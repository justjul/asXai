import time
import json
from confluent_kafka import Consumer
import threading
from pathlib import Path
import os
import hashlib

import config
from asxai.vectorDB import QdrantManager
from asxai.vectorDB import PaperEmbed, RerankEncoder
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

KAFKA_BOOTSTRAP = "kafka:9092" if running_inside_docker() else "localhost:29092"
SEARCH_REQ_TOPIC = "search-requests"
consumer = Consumer({'bootstrap.servers': KAFKA_BOOTSTRAP,
                     'group.id': 'search-group',
                     'auto.offset.reset': 'earliest'})
consumer.subscribe([SEARCH_REQ_TOPIC])

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


def run_search_worker():
    logger.info("Search worker ready")

    while True:
        payloads = []
        start = time.time()

        while len(payloads) < BATCH_SIZE and time.time() - start < MAX_WAIT:
            msg = consumer.poll(0.05)
            if msg and not msg.error():
                payload = json.loads(msg.value())
                payloads.append(payload)

        if not payloads:
            continue

        async_runner.run(batch_and_save(payloads))


async def batch_and_save(raw_payloads):
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

    # rerank_scores = {}
    # for i, qid in enumerate(query_ids):
    #     rerank_scores[qid] = rerankEngine.compute_max_sim(
    #         rerank_query_embeds[i], rerank_res_embeds[res_embeds_offset[i]:res_embeds_offset[i+1]]).cpu().tolist()

    rerank_scores = {}
    for i, qid in enumerate(query_ids):
        rerank_scores[qid] = rerankEngine.compute_max_sim(
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

        rerank_version_date = f"{rerankEngine.version}-{rerankEngine.training_date}"

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
                             "publicationDate_end": search_queries[qid].get("publicationDate_end")},
            "query_embedding": []  # query_embeds[i].tolist()
        }

        result_json = [{**base_json, **pl} for pl in payload]

        # Save to disk
        save_results(f"{task_ids[qid]}/{qid}", result_json)
        logger.info(f"Results for {qid} completed")


def save_results(search_id: str, result_data: list):
    users_root = config.USERS_ROOT
    search_path = os.path.join(users_root, f"{search_id}.json")
    os.makedirs(os.path.dirname(search_path), exist_ok=True)

    logger.info(f"Search path: {search_path}")

    # Save payload
    with open(search_path, "w") as f:
        json.dump(result_data, f)


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


def load_existing_result(task_id: str):
    users_root = config.USERS_ROOT
    full_path = os.path.join(users_root, f"{task_id}.inprogress")
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            try:
                return json.load(f)
            except Exception as e:
                logger.warning(f"Could not decode result for {task_id}: {e}")
                return []
    return []


def mark_as_inprogress(task_id: str):
    users_root = config.USERS_ROOT
    json_path = os.path.join(users_root, f"{task_id}.json")
    inprogress_path = os.path.join(users_root, f"{task_id}.inprogress")

    if os.path.exists(json_path):
        os.replace(json_path, inprogress_path)
    elif not os.path.exists(inprogress_path):
        os.makedirs(os.path.dirname(inprogress_path), exist_ok=True)
        with open(inprogress_path, "w") as f:
            json.dump([], f)


def mark_as_complete(task_id: str):
    users_root = config.USERS_ROOT
    inprogress_path = os.path.join(users_root, f"{task_id}.inprogress")
    json_path = os.path.join(users_root, f"{task_id}.json")

    if os.path.exists(inprogress_path):
        os.replace(inprogress_path, json_path)


def start_search_engine():
    t1 = threading.Thread(target=run_search_worker, daemon=True)

    t1.start()

    return t1


if __name__ == "__main__":
    t = threading.Thread(target=run_search_worker, daemon=True)

    t.start()
    t.join()
