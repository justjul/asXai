import time
import json
from confluent_kafka import Consumer
import threading
from pathlib import Path
import os

import config
from vectorDB import QdrantManager
from vectorDB import PaperEmbed, RerankEncoder
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import math

from typing import Union, List
from src.utils import AsyncRunner
import logging
from src.logger import get_logger
from src.utils import load_params

logger = get_logger(__name__, level=logging.INFO)

params = load_params()
search_config = params["search"]


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
    async_runner = AsyncRunner()
    embedEngine = PaperEmbed(model_name=config.MODEL_EMBED)
    qdrant = QdrantManager()
    rerankEngine = RerankEncoder.load()

    KAFKA_BOOTSTRAP = "localhost:29092"  # "kafka:9092"

    consumer = Consumer({'bootstrap.servers': KAFKA_BOOTSTRAP,
                        'group.id': 'search-group',
                         'auto.offset.reset': 'earliest'})
    consumer.subscribe(['search-requests'])

    BATCH_SIZE = 8
    MAX_WAIT = 0.2

    logger.info("Search worker ready")

    while True:
        queries, query_ids, raw_payloads = [], [], []
        start = time.time()

        while len(queries) < BATCH_SIZE and time.time() - start < MAX_WAIT:
            msg = consumer.poll(0.05)
            if msg and not msg.error():
                payload = json.loads(msg.value())
                queries.append(payload["query"])
                query_ids.append(payload["id"])
                raw_payloads.append(payload)

        logger.setLevel(logging.INFO)

        if not queries:
            continue

        query_embeds = embedEngine.embed_queries(queries)

        async def batch_and_save():
            results = await qdrant.query_batch_streamed(
                query_vectors=query_embeds.tolist(),
                query_ids=query_ids,
                topK=search_config["topk_qdrant"],
                collection_name=qdrant.collection_name_ids,
                with_vectors=True,
            )

            rerank_query_embeds = rerankEngine(
                query_embeds.unsqueeze(1).to(rerankEngine.device))
            res_embeds_offset = [
                0] + [len(results[qid].points) for qid in query_ids]
            res_embeds_offset = np.cumsum(res_embeds_offset)
            rerank_res_embeds = rerankEngine(
                [torch.tensor(pt.vector) for qid in query_ids for pt in results[qid].points])

            rerank_scores = {}
            for i, qid in enumerate(query_ids):
                rerank_scores[qid] = rerankEngine.compute_max_sim(
                    rerank_query_embeds[i], rerank_res_embeds[res_embeds_offset[i]:res_embeds_offset[i+1]]).cpu().tolist()

            for i, qid in enumerate(query_ids):
                payload = [pt.payload | {'qdrant_score': pt.score,
                                         'rerank_score': rerank_scores[qid][i],
                                         'user_score': None,
                                         }
                           for i, pt in enumerate(results[qid].points)]
                for pl in payload:
                    pl['main_text'] = []

                existing_result = load_existing_result(qid)
                rerank_version_date = f"{rerankEngine.version}-{rerankEngine.training_date}"

                if existing_result and existing_result.get("query", []) == queries[i]:
                    existing_payload = existing_result['result']
                    existing_ids = {item['paperId']
                                    for item in existing_payload if 'paperId' in item}

                    # replacing rerank_scores to make sure most payloads are
                    # reranked with the latest model
                    ids_to_rerank_score = {pl['paperId']: pl['rerank_score']
                                           for pl in payload
                                           if 'paperId' in pl}

                    for item in existing_payload:
                        paper_id = item.get('paperId')
                        # if item['rerank_score'] > 0.7:
                        #     item['user_score'] = 1
                        # else:
                        #     item['user_score'] = 0
                        if paper_id in ids_to_rerank_score:
                            item['rerank_score'] = ids_to_rerank_score[paper_id]

                    new_payload = [
                        pl for pl in payload if pl['paperId'] not in existing_ids]
                    new_payload = new_payload[:search_config["topk_rerank"]]
                    payload = existing_payload + new_payload

                blendscorer = BlendScorer.load(qid)
                trained = blendscorer.fit(payload)
                if trained:
                    blendscorer.save(qid)
                    logger.info(f"blendscorer trained and saved for {qid}")

                blended_scores = blendscorer.predict(payload)
                for pl, s in zip(payload, blended_scores):
                    pl["blended_score"] = s
                    pl["score"] = s if pl["user_score"] != 0 else 0

                payload = sorted(payload,
                                 key=lambda p: p['score'],
                                 reverse=True)
                result_json = {
                    "id": qid,
                    "task_id": qid,
                    "query": raw_payloads[i]["query"],
                    "user_id": raw_payloads[i].get("user_id"),
                    "notebook_id": raw_payloads[i].get("notebook_id"),
                    "qdrant_model": qdrant.model_name,
                    "rerank_model": rerank_version_date,
                    "blend_weights": blendscorer.weights,
                    "result": payload,
                }

                # Save to disk
                save_result(qid, json.dumps(result_json))
                logger.info(f"Saved results for {qid}")

        async_runner.run(batch_and_save())


def save_result(task_id: str, result_data: str):
    users_root = config.USERS_ROOT
    full_path = os.path.join(users_root, f"{task_id}.json")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(result_data)


def load_existing_result(task_id: str):
    users_root = config.USERS_ROOT
    full_path = os.path.join(users_root, f"{task_id}.json")
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def start_search_engine():
    t1 = threading.Thread(target=run_search_worker, daemon=True)

    t1.start()

    return t1


if __name__ == "__main__":
    t = threading.Thread(target=run_search_worker, daemon=True)

    t.start()
    t.join()
