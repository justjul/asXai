# Search API – collects user queries, submits them to Kafka, and streams back Qdrant results
# -----------------------------------------------------------------------------
#   POST /search                 → returns {"task_id": <uuid>}
#   GET  /search/stream/{task_id}  (SSE) → incremental search hits as they arrive
# -----------------------------------------------------------------------------

import json
from uuid import uuid4
from typing import Optional, List
import os

from fastapi import FastAPI,  HTTPException
import hashlib
from pydantic import BaseModel
from confluent_kafka import Producer

import config
from src.utils import load_params

params = load_params()
search_config = params["search"]

KAFKA_BOOTSTRAP = "localhost:29092"  # "kafka:9092"
SEARCH_REQ_TOPIC = "search-requests"
HASH_LEN = search_config['hash_len']

USERS_ROOT = config.USERS_ROOT

producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
app = FastAPI(title="Search API")


# Utilities
def hash_id(value: str, length: int = 20) -> str:
    full = hashlib.sha256(value.strip().encode()).hexdigest()
    return full[:length]


def make_task_id(user_id: str, notebook_id: str) -> str:
    return f"{hash_id(user_id, length=HASH_LEN)}/{hash_id(notebook_id, length=HASH_LEN)}"


def persist_path(task_id: str) -> str:
    full_path = os.path.join(USERS_ROOT, f"{task_id}.json")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path


def get_persisted_result(task_id: str) -> Optional[dict]:
    path = persist_path(task_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def list_notebooks(user_id: str) -> List[str]:
    user_hash = hash_id(user_id)
    user_dir = os.path.join(USERS_ROOT, user_hash)
    if not os.path.exists(user_dir):
        return []

    notebooks = []
    for f in os.listdir(user_dir):
        if f.endswith(".json"):
            try:
                with open(os.path.join(user_dir, f), "r") as j:
                    data = json.load(j)
                    notebooks.append({"id": data.get("notebook_id", f[:-5]),
                                      "title": data.get("query", f[:-5])})
            except Exception:
                continue
    return notebooks


# Query model
class QueryRequest(BaseModel):
    user_id: str
    notebook_id: str
    query: str


@app.post("/search")
async def create_search(req: QueryRequest):
    task_id = make_task_id(req.user_id, req.notebook_id)
    payload = {
        "id": task_id,
        "query": req.query,
        "user_id": req.user_id,
        "notebook_id": req.notebook_id
    }
    producer.produce(SEARCH_REQ_TOPIC, key=task_id, value=json.dumps(payload))
    producer.flush()
    return {"task_id": task_id}


@app.get("/search/result/{task_id:path}")
async def get_latest_result(task_id: str):
    result = get_persisted_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="No result found for this task_id")
    return {"task_id": task_id, "result": result}


@app.get("/user/{user_id}/notebooks")
async def get_user_notebooks(user_id: str):
    notebooks = list_notebooks(user_id)
    return {"user_id": user_id, "notebooks": notebooks}


@app.get("/user/{user_id}/notebooks/{notebook_id}")
async def get_notebook_result(user_id: str, notebook_id: str):
    task_id = make_task_id(user_id, notebook_id)
    result = get_persisted_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="No result found for this notebook")
    return {"task_id": task_id, "result": result}


@app.get("/task_id")
def generate_task_id(user_id: str, notebook_id: str):
    return {"task_id": make_task_id(user_id, notebook_id)}
