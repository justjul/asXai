# Search API – collects user queries, submits them to Kafka, and send back Qdrant results

import json
from uuid import uuid4
from typing import Optional, List
import os
import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response
import uvicorn
import hashlib
from pydantic import BaseModel
from confluent_kafka import Producer

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry

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


_custom_registry = CollectorRegistry()

# PROMETHEUS metrics
REQUEST_COUNT = Counter(
    "search_api_requests_total",
    "Total number of HTTP requests to the search API",
    ["method", "endpoint", "http_status"],
    registry=_custom_registry,
)

REQUEST_LATENCY = Histogram(
    "search_api_request_latency_seconds",
    "Histogram of request latency (seconds) for the search API",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    registry=_custom_registry,
)

IN_PROGRESS = Gauge(
    "search_api_requests_in_progress",
    "Number of in-progress HTTP requests to the search API",
    registry=_custom_registry,
)


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


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path

    IN_PROGRESS.inc()                  # track one more in-flight request
    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        latency = time.time() - start_time

        REQUEST_LATENCY.labels(
            method=method, endpoint=endpoint).observe(latency)
        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, http_status=status_code
        ).inc()
        IN_PROGRESS.dec()              # request done, decrement

    return response


# Utilities
def hash_id(value: str, length: int = 20) -> str:
    full = hashlib.sha256(value.strip().encode()).hexdigest()
    return full[:length]


def make_task_id(user_id: str, notebook_id: str) -> str:
    return f"{user_id}/{notebook_id}"


def result_path(task_id: str, inprogress: bool = False) -> str:
    ext = ".inprogress" if inprogress else ".json"
    full_path = os.path.join(USERS_ROOT, f"{task_id}{ext}")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path


def get_result(task_id: str, timeout: float = search_config['timeout']) -> Optional[dict]:
    json_path = result_path(task_id)
    inprogress_path = result_path(task_id, inprogress=True)
    time.sleep(0.5)
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
    query_id: str
    query: dict
    topK: int = search_config["topk_rerank"]
    paperLock: bool = False


@app.post("/search")
async def create_search(req: QueryRequest):
    task_id = make_task_id(req.user_id, req.notebook_id)
    payload = {
        "task_id": task_id,
        "query": req.query,
        "user_id": req.user_id,
        "query_id": req.query_id,
        "notebook_id": req.notebook_id,
        "topK": req.topK,
        "paperLock": req.paperLock
    }
    print(payload)
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


@app.get("/metrics")
def metrics():
    data = generate_latest(_custom_registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("asxai.services.search.search_api:app",
                host="0.0.0.0", port=8000, reload=False)
