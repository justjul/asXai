import time
import json
from confluent_kafka import Consumer
from confluent_kafka.admin import AdminClient, NewTopic
import threading
import asyncio
from pathlib import Path
import os
import hashlib

import config
from asxai.vectorDB import QdrantManager
from asxai.vectorDB import PaperEmbed, RerankEncoder, InnovEncoder, compute_max_sim
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from typing import Union, List
from asxai.utils import AsyncRunner
from asxai.logger import get_logger
from asxai.utils import load_params, running_inside_docker

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
search_config = params["search"]

async_runner = AsyncRunner()
embedEngine = PaperEmbed(model_name=config.MODEL_EMBED)
qdrant = QdrantManager()
rerankEngine = RerankEncoder.load()
if not rerankEngine:
    rerankEngine = RerankEncoder()
innovEngine = InnovEncoder.load()
if not innovEngine:
    innovEngine = InnovEncoder()


search_queue = asyncio.Queue()
question_queue = asyncio.Queue()


def kafka_poller(consumer, queue: asyncio.Queue):
    while True:
        msg = consumer.poll(1.0)
        if msg and not msg.error():
            payload = json.loads(msg.value())
            async_runner.run(queue.put(payload))


def create_topic_if_needed(bootstrap_servers: str, topic_name: str, num_partitions: int = 1, replication_factor: int = 1):
    admin_client = AdminClient({'bootstrap.servers': bootstrap_servers})
    topic_metadata = admin_client.list_topics(timeout=5)

    if topic_name not in topic_metadata.topics:
        logger.info(f"Creating Kafka topic '{topic_name}'...")
        new_topic = NewTopic(
            topic=topic_name, num_partitions=num_partitions, replication_factor=replication_factor)
        fs = admin_client.create_topics([new_topic])

        for topic, f in fs.items():
            try:
                f.result()
                logger.info(f"Topic '{topic}' created successfully.")
            except Exception as e:
                logger.error(f"Failed to create topic '{topic}': {e}")
    else:
        logger.info(f"Kafka topic '{topic_name}' already exists.")


BATCH_SIZE = 8
MAX_WAIT = 0.2


def hash_query(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


class BlendScorer:
    def __init__(self):
        self.model = LogisticRegression(penalty='l2',
                                        solver='liblinear',
                                        class_weight='balanced')
        self.n_examples = 0
        self.weights = {}

    def fit(self, payloads):
        X, y = [], []
        for p in payloads:
            if p['user_score'] is not None:
                X.append([p["qdrant_score"],
                          p["rerank_score"]])
                y.append(int(p["user_score"]))

        if len(X) < 30 or len(X) == self.n_examples:
            return False

        self.model.fit(X, y)
        print(self.model.classes_)
        self.n_examples = len(X)
        weights = self.model.coef_[0]
        feature_names = ["qdrant_score",
                         "rerank_score"]
        self.weights = dict(zip(feature_names, weights))
        return True

    def predict(self, payloads):
        n_feedback = len([p["user_score"]
                         for p in payloads if p['user_score'] is not None])
        try:
            if n_feedback > 30:
                X = [[p["qdrant_score"], p["rerank_score"]] for p in payloads]
                blend_scores = self.model.predict_proba(X)[:, 1].tolist()
                return blend_scores
        except Exception as e:
            logger.info(f"blendscorer likely not fitted yet: {e}")

        return [p["rerank_score"] for p in payloads]

    def save(self, qid):
        path = os.path.join(config.USERS_ROOT, qid + "_blend_model.pkl")
        joblib.dump({"model": self.model,
                     "n_examples": self.n_examples,
                     "weights": self.weights,
                     }, path)

    @classmethod
    def load(cls, qid):
        path = os.path.join(config.USERS_ROOT, qid + "_blend_model.pkl")
        if os.path.exists(path):
            state = joblib.load(path)
            instance = cls()
            instance.model = state["model"]
            instance.n_examples = state.get("n_examples", 0)
            instance.weights = state.get("weights", {})
            return instance
        return cls()


async def search_process(raw_payloads):
    print(raw_payloads)
    query_ids = {pl["query_id"]
                 for pl in raw_payloads}
    search_queries = {pl["query_id"]: pl["query"] for pl in raw_payloads}
    task_ids = {pl["query_id"]: pl["task_id"]
                for pl in raw_payloads}
    topKs = {pl["query_id"]: pl["topK"]
             for pl in raw_payloads}
    paperLocks = {pl["query_id"]: pl["paperLock"]
                  for pl in raw_payloads}

    # expanding query_ids and queries for batch embeddings
    expanded_query_ids = [
        qid for qid in query_ids for _ in search_queries[qid].get('queries')
    ]
    expanded_queries = [
        q for qid in query_ids for q in search_queries[qid].get('queries')
    ]
    q_embeds = embedEngine.embed_queries(expanded_queries)

    r_embeds = rerankEngine(q_embeds.unsqueeze(1).to(rerankEngine.device))

    # Reconstructing list of embeddings per query_id
    query_embeds, rerank_embeds = {}, {}
    for qid, q_emb, r_emb in zip(expanded_query_ids, q_embeds.tolist(), r_embeds):
        query_embeds.setdefault(qid, []).append(q_emb)
        rerank_embeds.setdefault(qid, []).append(r_emb)

    meta_filters = []
    for qid in query_ids:
        meta_filters.append([])
        if search_queries[qid].get('authorName', None):
            meta_filters[-1].append(['authorName',
                                    '==', search_queries[qid]['authorName'].split(',')])
        if search_queries[qid].get('publicationDate_start', None):
            meta_filters[-1].append(['publicationDate', 'gte',
                                    search_queries[qid]['publicationDate_start']])
        if search_queries[qid].get('publicationDate_end', None):
            meta_filters[-1].append(['publicationDate', 'lte',
                                    search_queries[qid]['publicationDate_end']])

        if search_queries[qid].get('search_paperIds', None):
            meta_filters[-1].append(['paperId', '==',
                                    search_queries[qid]['search_paperIds']])
        if search_queries[qid].get('peer_reviewed_only', False):
            meta_filters[-1].append(['venue', '!=', ['arxiv.org', 'bioRxiv']])
        if search_queries[qid].get('preprint_only', False):
            meta_filters[-1].append(['venue', '==', ['arxiv.org', 'bioRxiv']])
        if search_queries[qid].get('venues', []):
            meta_filters[-1].append(['venue', '==',
                                    search_queries[qid].get('venues')])
        if search_queries[qid].get('citationCount', 0):
            meta_filters[-1].append(['citationCount', 'gte',
                                    search_queries[qid].get('citationCount')])

        existing_results = load_existing_result(task_ids[qid])
        if paperLocks[qid]:
            existing_paper_ids = [
                pl.get('paperId') for pl in existing_results if pl.get('user_score') >= 0]
            meta_filters[-1].append(['paperId', '==', existing_paper_ids])
            logger.info([pl.get('paperId') for pl in existing_results])

        trashed_paper_ids = [
            pl.get('paperId') for pl in existing_results if pl.get('user_score') < 0]
        meta_filters[-1].append(['paperId', '!=', trashed_paper_ids])
        logger.info([pl.get('paperId') for pl in trashed_paper_ids])

    results = await qdrant.query_batch_streamed(
        query_vectors=[query_embeds[qid] for qid in query_ids],
        query_ids=query_ids,
        topKs=[search_config["ntopk_qdrant"]*topKs[qid] for qid in query_ids],
        topK_per_paper=search_config["topk_per_article"],
        payload_filters=meta_filters,
        with_vectors=True,
    )

    empty_ids = [qid for qid in query_ids
                 if not results.get(qid) or not results[qid].points]
    query_ids = [qid for qid in query_ids
                 if qid not in empty_ids]

    if empty_ids:
        for qid in empty_ids:
            save_results(f"{task_ids[qid]}/{qid}", [{"query_id": qid}])
        logger.warning(
            f"Skipping reranking for {len(empty_ids)} queries with no Qdrant results: {empty_ids}")

    if not query_ids:
        logger.warning(
            "No valid task_ids remaining after filtering empty Qdrant results. Exiting.")
        return

    # query_embeds = torch.stack([query_embeds[i] for i, qid in enumerate(query_ids)
    #                             if qid not in empty_ids])
    raw_payloads = [raw_payloads[i] for i, qid in enumerate(query_ids)
                    if qid not in empty_ids]

    # rerank_query_embeds = rerankEngine(
    #     query_embeds.unsqueeze(1).to(rerankEngine.device))
    res_embeds_offset = [
        0] + [len(results[qid].points) for qid in query_ids]
    res_embeds_offset = np.cumsum(res_embeds_offset)
    rerank_res_embeds = rerankEngine(
        [torch.tensor(pt.vector) for qid in query_ids for pt in results[qid].points])

    rerank_scores = {}
    for i, qid in enumerate(query_ids):
        rerank_scores[qid] = compute_max_sim(
            torch.stack(rerank_embeds[qid], dim=1), rerank_res_embeds[res_embeds_offset[i]:res_embeds_offset[i+1]]).cpu().tolist()

    for i, qid in enumerate(query_ids):
        payload = [pt.payload | {'qdrant_score': pt.score / len(query_embeds[qid]),
                                 'rerank_score': rerank_scores[qid][k] / len(query_embeds[qid]),
                                 'user_score': 1,
                                 }
                   for k, pt in enumerate(results[qid].points)]
        for pl in payload:
            pl['main_text'] = []

        payload = payload[:topKs[qid]]

        if hasattr(rerankEngine, 'version'):
            rerank_version_date = f"{rerankEngine.version}-{rerankEngine.training_date}"
        else:
            rerank_version_date = 'None'

        blendscorer = BlendScorer.load(task_ids[qid])
        trained = blendscorer.fit(payload)
        if trained:
            blendscorer.save(task_ids[qid])
            logger.info(f"blendscorer trained and saved for {task_ids[qid]}")

        blended_scores = blendscorer.predict(payload)
        for pl, s in zip(payload, blended_scores):
            pl["blended_score"] = s
            pl["score"] = s if pl["user_score"] != 0 else 0

        payload = sorted(payload,
                         key=lambda p: p['score'],
                         reverse=True)
        base_json = {
            "id": qid,
            "task_id": qid,
            "query": raw_payloads[i]["query"],
            "query_id": qid,
            "user_id": raw_payloads[i].get("user_id"),
            "notebook_id": raw_payloads[i].get("notebook_id"),
            "qdrant_model": qdrant.model_name,
            "rerank_model": rerank_version_date,
            "blend_weights": blendscorer.weights,
            "timestamp": time.time(),
            "parsed_query": {"query": search_queries[qid].get("query"),
                             "authorName": search_queries[qid].get("authorName"),
                             "publicationDate_start": search_queries[qid].get("publicationDate_start"),
                             "publicationDate_end": search_queries[qid].get("publicationDate_end"),
                             "peer_reviewed_only": search_queries[qid].get("query", False),
                             "preprint_only": search_queries[qid].get("query", False),
                             "venues": search_queries[qid].get("venues", []), },
            "query_embedding": []  # query_embeds[i].tolist()
        }

        result_json = [{**base_json, **pl} for pl in payload]

        # Save to disk
        save_results(f"{task_ids[qid]}/{qid}", result_json)
        logger.info(f"Results for {qid} completed")


def save_results(search_id: str, result_data: list):
    search_path = config.USERS_ROOT / f"{search_id}.json"
    inprogress_path = config.USERS_ROOT / f"{search_id}.inprogress"
    os.makedirs(os.path.dirname(search_path), exist_ok=True)

    logger.info(f"Search path: {search_path}")

    # Save payload
    with open(inprogress_path, "w") as f:
        json.dump(result_data, f)

    os.rename(inprogress_path, search_path)


def load_existing_result(task_id: str):
    full_path = config.USERS_ROOT / f"{task_id}.json"
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            try:
                return json.load(f)
            except Exception as e:
                logger.warning(f"Could not decode result for {task_id}: {e}")
                return []
    return []


def append_results(task_id: str, result_data: list):
    users_root = config.USERS_ROOT
    full_path = os.path.join(users_root, f"{task_id}.inprogress")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        try:
            with open(full_path, "r") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except (json.JSONDecodeError, IOError):
            existing_data = []
    else:
        existing_data = []

    logger.info(f"existing search data: {existing_data}")

    # Append new result
    existing_data.extend(result_data)

    # Save updated list
    with open(full_path, "w") as f:
        json.dump(existing_data, f)


def mark_as_inprogress(task_id: str):
    json_path = config.USERS_ROOT / f"{task_id}.json"
    inprogress_path = config.USERS_ROOT / f"{task_id}.inprogress"

    if os.path.exists(json_path):
        os.replace(json_path, inprogress_path)
    elif not os.path.exists(inprogress_path):
        os.makedirs(os.path.dirname(inprogress_path), exist_ok=True)
        with open(inprogress_path, "w") as f:
            json.dump([], f)


def mark_as_complete(task_id: str):
    inprogress_path = config.USERS_ROOT / f"{task_id}.inprogress"
    json_path = config.USERS_ROOT / f"{task_id}.json"

    if os.path.exists(inprogress_path):
        os.replace(inprogress_path, json_path)


async def question_process(raw_payloads):
    print(raw_payloads)
    query_ids = {pl["query_id"]
                 for pl in raw_payloads}
    search_queries = {pl["query_id"]: pl["query"] for pl in raw_payloads}
    task_ids = {pl["query_id"]: pl["task_id"]
                for pl in raw_payloads}

    # expanding query_ids and queries for batch embeddings
    expanded_query_ids = [
        qid for qid in query_ids for _ in search_queries[qid].get('queries')
    ]
    expanded_queries = [
        q for qid in query_ids for q in search_queries[qid].get('queries')
    ]
    q_embeds = embedEngine.embed_queries(expanded_queries)

    # Reconstructing list of embeddings per query_id
    query_embeds = {}
    for qid, q_emb in zip(expanded_query_ids, q_embeds.tolist()):
        query_embeds.setdefault(qid, []).append(q_emb)

    meta_filters = []
    for qid in query_ids:
        meta_filters.append([])
        existing_results = load_existing_result(task_ids[qid])

        existing_paper_ids = [
            pl.get('paperId') for pl in existing_results if pl.get('user_score') >= 0]
        meta_filters[-1].append(['paperId', '==', existing_paper_ids])
        logger.info([pl.get('paperId') for pl in existing_results])

        trashed_paper_ids = [
            pl.get('paperId') for pl in existing_results if pl.get('user_score') < 0]
        meta_filters[-1].append(['paperId', '!=', trashed_paper_ids])
        logger.info([pl.get('paperId') for pl in trashed_paper_ids])

    results = await qdrant.query_batch_streamed(
        query_vectors=[query_embeds[qid] for qid in query_ids],
        query_ids=query_ids,
        topKs=[len(existing_paper_ids) for qid in query_ids],
        topK_per_paper=search_config["topk_per_article"],
        payload_filters=meta_filters,
        with_vectors=True,
    )

    # rerank_query_embeds = rerankEngine(
    #     query_embeds.unsqueeze(1).to(rerankEngine.device))
    res_embeds_offset = [
        0] + [len(results[qid].points) for qid in query_ids]
    res_embeds_offset = np.cumsum(res_embeds_offset)
    innov_res_embeds = innovEngine(
        [torch.tensor(pt.vector) for qid in query_ids for pt in results[qid].points])

    # Need to transform query_embeds to tensors
    payloads = {}
    for i, qid in enumerate(query_ids):
        payloads[qid] = []
        n_papers = res_embeds_offset[i+1] - res_embeds_offset[i]
        for k, q_emb in enumerate(query_embeds[qid]):
            query = search_queries[qid].get('queries')[k]
            q_emb = torch.tensor(q_emb).to(innovEngine.device)
            q_emb = q_emb.unsqueeze(0).unsqueeze(1)
            q_emb = q_emb.expand(n_papers, -1, -1)
            innov_embs = innov_res_embeds[res_embeds_offset[i]:res_embeds_offset[i+1]]
            innov_embs = innov_embs.unsqueeze(1)
            innov_sim = compute_max_sim(innov_embs, q_emb).cpu().tolist()
            payloads[qid].append(
                {'query': query, 'score': float(np.mean(innov_sim))})

        print(payloads[qid])
        payloads[qid] = sorted(
            payloads[qid], key=lambda p: p['score'], reverse=True)

    for qid, payload in payloads.items():
        if hasattr(innovEngine, "version"):
            innov_version_date = f"{innovEngine.version}-{innovEngine.training_date}"
        else:
            innov_version_date = 'None'

        base_json = {
            "id": qid,
            "task_id": qid,
            "query_id": qid,
            "user_id": raw_payloads[i].get("user_id"),
            "notebook_id": raw_payloads[i].get("notebook_id"),
            "qdrant_model": qdrant.model_name,
            "innov_model": innov_version_date,
            "timestamp": time.time(),
        }

        result_json = [{**base_json, **pl} for pl in payload]

        # Save to disk
        save_innov_results(f"{task_ids[qid]}", result_json)
        logger.info(f"Open questions search for {qid} completed")


def save_innov_results(task_id: str, result_data: list):
    innov_path = config.USERS_ROOT / f"{task_id}.innov.json"
    inprogress_path = config.USERS_ROOT / f"{task_id}.innov.inprogress"
    os.makedirs(os.path.dirname(innov_path), exist_ok=True)

    logger.info(f"Innov path: {innov_path}")

    # Save payload
    with open(inprogress_path, "w") as f:
        json.dump(result_data, f)

    os.rename(inprogress_path, innov_path)


async def search_worker_loop():
    while True:
        payloads = []
        start = time.time()

        while len(payloads) < BATCH_SIZE and time.time() - start < MAX_WAIT:
            payload = await search_queue.get()
            payloads.append(payload)
            print("payload found")

        if not payloads:
            continue

        asyncio.create_task(search_process(payloads))


async def question_worker_loop():
    while True:
        payloads = []
        start = time.time()

        while len(payloads) < BATCH_SIZE and time.time() - start < MAX_WAIT:
            payload = await question_queue.get()
            payloads.append(payload)
            print("payload found")

        if not payloads:
            continue

        asyncio.create_task(question_process(payloads))


async def search_loops():
    logger.info("Search and Question workers running")
    await asyncio.gather(
        search_worker_loop(),
        question_worker_loop()
    )


def run_search_worker():
    KAFKA_BOOTSTRAP = "kafka:9092" if running_inside_docker() else "localhost:29092"
    SEARCH_REQ_TOPIC = "search-requests"
    QUESTIONS_REQ_TOPIC = "questions-requests"

    create_topic_if_needed(KAFKA_BOOTSTRAP, SEARCH_REQ_TOPIC)
    consumer = Consumer({'bootstrap.servers': KAFKA_BOOTSTRAP,
                        'group.id': 'search-group',
                         'auto.offset.reset': 'earliest'})
    consumer.subscribe([SEARCH_REQ_TOPIC])

    create_topic_if_needed(KAFKA_BOOTSTRAP, QUESTIONS_REQ_TOPIC)
    consumer_Q = Consumer({'bootstrap.servers': KAFKA_BOOTSTRAP,
                           'group.id': 'question-group',
                           'auto.offset.reset': 'earliest'})
    consumer_Q.subscribe([QUESTIONS_REQ_TOPIC])

    threading.Thread(target=kafka_poller,
                     args=(consumer, search_queue),
                     daemon=True).start()

    threading.Thread(target=kafka_poller,
                     args=(consumer_Q, question_queue),
                     daemon=True).start()

    async_runner.run(search_loops())


def start_search_engine():
    t1 = threading.Thread(target=run_search_worker, daemon=True)

    t1.start()

    return t1


if __name__ == "__main__":
    async_runner.run(run_search_worker())
    # t1 = threading.Thread(target=run_search_worker, daemon=True)

    # t1.start()
    # t1.join()
