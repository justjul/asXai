# Search API – collects user queries, submits them to Kafka, and send back Qdrant results

import json
from uuid import uuid4
from typing import Optional, List
import os
import time

from fastapi import FastAPI,  HTTPException
import uvicorn
import hashlib
from pydantic import BaseModel
from confluent_kafka import Producer

import config
from asxai.logger import get_logger
from asxai.utils import load_params, running_inside_docker

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
search_config = params["search"]

KAFKA_BOOTSTRAP = "kafka:9092" if running_inside_docker() else "localhost:29092"
SEARCH_REQ_TOPIC = "search-requests"
HASH_LEN = search_config['hash_len']

USERS_ROOT = config.USERS_ROOT


def wait_for_kafka(bootstrap_servers, retries=10, delay=3):
    for attempt in range(retries):
        try:
            p = Producer({"bootstrap.servers": bootstrap_servers})
            p.list_topics(timeout=5)
            print("Kafka broker up and ready")
            return p
        except Exception as e:
            print(f"[Kafka] Waiting for broker ({attempt+1}/{retries})... {e}")
            time.sleep(delay)
    raise RuntimeError("Kafka broker not available")


KAFKA_BOOTSTRAP = "kafka:9092" if running_inside_docker() else "localhost:29092"
producer = wait_for_kafka(KAFKA_BOOTSTRAP)

# producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
app = FastAPI(title="Search API")


# Utilities
def hash_id(value: str, length: int = 20) -> str:
    full = hashlib.sha256(value.strip().encode()).hexdigest()
    return full[:length]


def make_task_id(user_id: str, notebook_id: str) -> str:
    return f"{hash_id(user_id, length=HASH_LEN)}/{hash_id(notebook_id, length=HASH_LEN)}"


def result_path(task_id: str, inprogress: bool = False) -> str:
    ext = ".inprogress" if inprogress else ".json"
    full_path = os.path.join(USERS_ROOT, f"{task_id}{ext}")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path


def get_result(task_id: str, timeout: float = search_config['timeout']) -> Optional[dict]:
    json_path = result_path(task_id)
    inprogress_path = result_path(task_id, inprogress=True)
    start = time.time()

    # Wait for .json file
    while not os.path.exists(json_path):
        if time.time() - start > timeout:
            logger.info(
                f"Result for {task_id} not found after {timeout}s. Will try loading in-progress result.")
            break
        time.sleep(0.05)

    path_to_load = json_path if os.path.exists(json_path) else inprogress_path
    if not os.path.exists(path_to_load):
        logger.warning(
            f"No result found for task_id={task_id} in either .json or .inprogress")
        return None

    try:
        with open(path_to_load) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(
            f"Could not decode result file for task_id={task_id}: {e}")
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


@app.get("/search/{task_id:path}")
async def get_latest_result(task_id: str):
    result = get_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"No result found for {task_id}")
    return {"task_id": task_id, "notebook": result}


@app.get("/search/{user_id}")
async def get_user_notebooks(user_id: str):
    notebooks = list_notebooks(user_id)
    return {"user_id": user_id, "notebook_list": notebooks}


@app.get("/search/{user_id}/{notebook_id}")
async def get_notebook_result(user_id: str, notebook_id: str):
    task_id = make_task_id(user_id, notebook_id)
    result = get_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="No result found for this notebook")
    return {"task_id": task_id, "notebook": result}


@app.get("/search/task_id")
def generate_task_id(user_id: str, notebook_id: str):
    return {"task_id": make_task_id(user_id, notebook_id)}


if __name__ == "__main__":
    uvicorn.run("asxai.services.search.search_api:app",
                host="0.0.0.0", port=8000, reload=False)
