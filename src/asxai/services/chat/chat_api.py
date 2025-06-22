# Notebook API – collects chat prompts, notebook results and send back Ollama answers

import shutil
from confluent_kafka.admin import AdminClient, NewTopic
import json
from uuid import uuid4
from typing import Optional, List
import os
import time
import re

from fastapi import FastAPI,  Request, HTTPException
import uvicorn
import hashlib
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from confluent_kafka import Producer, Consumer

import config
from asxai.logger import get_logger
from asxai.utils import load_params, running_inside_docker
from ..authenticate.auth import verify_token, set_admin_claim, revoke_admin_claim, get_firebase_users
from .notebook_manager import NotebookManager
from fastapi import Depends
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, Summary

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
chat_config = params["chat"]
search_config = params["search"]

KAFKA_BOOTSTRAP = "kafka:9092" if running_inside_docker() else "localhost:29092"
CHAT_REQ_TOPIC = "notebook-requests"
CHAT_CANCEL_TOPIC = "notebook-cancel"

HASH_LEN = search_config['hash_len']

USERS_ROOT = config.USERS_ROOT

_custom_registry = CollectorRegistry()

# PROMETHEUS metrics
REQUEST_COUNT = Counter(
    "chat_api_requests_total",
    "Total number of HTTP requests to the chat API",
    ["method", "endpoint", "http_status"],
    registry=_custom_registry,
)

REQUEST_LATENCY = Histogram(
    "chat_api_request_latency_seconds",
    "Histogram of request latency (seconds) for the chat API",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    registry=_custom_registry,
)

IN_PROGRESS = Gauge(
    "chat_api_requests_in_progress",
    "Number of in-progress HTTP requests to the chat API",
    registry=_custom_registry,
)

STREAM_FIRST_TOKEN = Summary(
    "stream_first_token_seconds",
    "Time to first token",
    registry=_custom_registry,
)
STREAM_TOTAL_DURATION = Summary(
    "stream_total_duration_seconds",
    "Total stream duration",
    registry=_custom_registry,
)


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

create_topic_if_needed(KAFKA_BOOTSTRAP, "notebook-queries")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code before the yield runs once, before the first request:
    set_admin_claim()  # grant admin=True to every UID in ADMIN_UIDS
    yield
    # Code after the yield would run on application shutdown

app = FastAPI(title="Chat API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=chat_config["allowed_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sanitize_endpoint(path):
    # Replace notebook IDs with :id for grouping
    return re.sub(r'/notebook/[^/]+', '/notebook/:id', path)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = sanitize_endpoint(request.url.path)

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


def create_user(user_id: str) -> str:
    user_path = os.path.join(USERS_ROOT, f"{user_id}")
    print(user_path)
    os.makedirs(user_path, exist_ok=True)
    return {"user_id": user_id}


def create_task_id(user_id: str) -> str:
    notebook_id = hash_id(user_id + str(time.time()), length=HASH_LEN)
    new_task_id = {"user_id": user_id,
                   "notebook_id": notebook_id,
                   "task_id": f"{user_id}/{notebook_id}",
                   }
    user_path = os.path.join(USERS_ROOT, f"{user_id}")
    print(user_path)
    os.makedirs(user_path, exist_ok=True)
    return new_task_id


def make_id(name_id: str) -> str:
    return f"{hash_id(name_id, length=HASH_LEN)}"


def safe_user_id(uid: str) -> str:
    return f"{hash_id(uid, length=HASH_LEN)}"


def result_path(task_id: str, inprogress: bool = False) -> str:
    ext = ".inprogress" if inprogress else ".json"
    full_path = os.path.join(USERS_ROOT, f"{task_id}{ext}")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path


def delete_user(user_id: str) -> str:
    user_path = os.path.join(USERS_ROOT, f"{user_id}")
    if os.path.isdir(user_path):
        shutil.rmtree(user_path)
        return {"user_id": user_id,
                "status": 'deleted',
                }
    return {}


def list_users():
    return os.listdir(USERS_ROOT)


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


Notebook_manager = NotebookManager(config.USERS_ROOT)


class ChatRequest(BaseModel):
    message: str
    query_id: Optional[str] = None
    search_query: Optional[dict] = None
    model: str = "default"
    topK: int = search_config["topk_rerank"]
    paperLock: bool = False
    mode: str = "reply"


class ExpandRequest(BaseModel):
    query_id: str
    model: str = "default"
    topK: int = search_config["topk_rerank"]
    paperLock: bool = False
    mode: str = "reply"


@app.post("/notebook/{notebook_id}/chat")
async def submit_chat(notebook_id: str, req: ChatRequest, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    create_user(user_id)
    task_id = f"{user_id}/{notebook_id}"
    query_id = req.query_id or hash_id(
        notebook_id + req.message + str(time.time()))
    payload = {
        "task_id": task_id,
        "query_id": query_id,
        "user_id": user_id,
        "notebook_id": notebook_id,
        "content": req.message,
        "search_query": req.search_query,
        "role": "user",
        "model": req.model,
        "topK": req.topK,
        "paperLock": req.paperLock,
        "mode": req.mode,
        "timestamp": time.time(),
        "done": 0
    }

    print(req.model)
    producer.produce(CHAT_REQ_TOPIC, key=task_id, value=json.dumps(payload))
    producer.flush()
    return {"status": "submitted",
            "task_id": task_id,
            "query_id": query_id,
            "user_id": user_id,
            "notebook_id": notebook_id}


@app.get("/notebook/{notebook_id}/chat/stream")
def stream_response(notebook_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    def event_stream():
        starttime = time.time()
        first_token_sent = False

        stream_path = os.path.join(config.USERS_ROOT, f"{task_id}.stream")

        # Wait for the file to be created
        endtime = time.time() + 60
        while not os.path.exists(stream_path):
            time.sleep(0.1)
            if time.time() > endtime:
                raise HTTPException(
                    status_code=404, detail="Chat stream not found.")

        try:
            with open(stream_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue

                    line = line.strip()

                    if line == "<SEARCHING>":
                        yield f"data: ------------------\n\n"
                        continue

                    yield f"data: {line}\n\n"

                    if line == "<END_OF_MESSAGE>":
                        break

                    if not first_token_sent:
                        STREAM_FIRST_TOKEN.observe(time.time() - starttime)
                        first_token_sent = True
        finally:
            STREAM_TOTAL_DURATION.observe(time.time() - starttime)
            # Cleanup the stream file after completion
            try:
                os.remove(stream_path)
            except Exception as e:
                logger.warning(
                    f"Could not delete stream file {stream_path}: {e}")

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/notebook/{notebook_id}/abort")
async def abort_generation(notebook_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"
    payload = {
        "task_id": task_id,
        "timestamp": time.time(),
    }
    producer.produce(CHAT_CANCEL_TOPIC, key=task_id, value=json.dumps(payload))
    producer.flush()
    return JSONResponse(content={"status": "cancelled", "task_id": task_id})


@app.post("/notebook/update")
async def update_all_notebooks(decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    create_user(user_id)

    # Get list of notebooks
    notebook_list = await Notebook_manager.list(user_id)
    if not notebook_list:
        raise HTTPException(status_code=404, detail="No notebooks found")

    content = "*notebook update*"
    # Update each notebook
    for notebook in notebook_list[0:1]:
        notebook_id = notebook["id"]
        print(notebook_id)
        query_id = hash_id(notebook_id + content + str(time.time()))
        task_id = f"{user_id}/{notebook_id}"

        payload = {
            "task_id": task_id,
            "query_id": query_id,
            "user_id": user_id,
            "notebook_id": notebook_id,
            "content": content,
            "search_query": None,
            "role": "user",
            "model": "default",
            "topK": None,
            "paperLock": False,
            "mode": "reply",
            "timestamp": time.time(),
            "done": 0
        }

        producer.produce(CHAT_REQ_TOPIC, key=task_id,
                         value=json.dumps(payload))

    producer.flush()

    return JSONResponse(content={"status": "submitted", "notebooks_updated": len(notebook_list)})


@app.get("/notebook/{notebook_id}/chat/final")
async def get_final_chat(notebook_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    chat_path = os.path.join(config.USERS_ROOT, f"{task_id}.chat.json")
    if not os.path.exists(chat_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")

    try:
        history = await Notebook_manager.load_history(task_id)

        endtime = time.time() + 10  # timeout after 10 seconds
        user_msg, assistant_msg = None, None
        query_id = None

        while (user_msg is None or assistant_msg is None) and time.time() < endtime:
            user_entry = next((m for m in reversed(history)
                              if m["role"] == "user"), None)
            if user_entry:
                user_msg = user_entry["content"]
                papers = user_entry["papers"]
                query_id = user_entry["query_id"]
            if query_id:
                assistant_msg = next(
                    (m["content"] for m in reversed(history)
                     if m["role"] == "assistant" and m.get("query_id") == query_id),
                    None
                )
                papers = papers or next(
                    (m["papers"] for m in reversed(history)
                     if m["role"] == "assistant" and m.get("query_id") == query_id),
                    None
                )
            if user_msg and assistant_msg:
                break
            time.sleep(0.1)  # avoid busy waiting

        if not user_msg or not assistant_msg:
            raise HTTPException(
                status_code=404, detail="Incomplete conversation.")

        return JSONResponse(content={"user": user_msg, "assistant": assistant_msg, "papers": papers})

    except Exception as e:
        logger.error(f"Error reading chat history for {task_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to read chat history.")


@app.get("/notebook/{notebook_id}/content/{query_id}")
async def get_chat_msg(notebook_id: str, query_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    chat_path = os.path.join(config.USERS_ROOT, f"{task_id}.chat.json")
    if not os.path.exists(chat_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")

    try:
        query_msg = await Notebook_manager.load_query(task_id, query_id)

        if not query_msg:
            raise HTTPException(
                status_code=404, detail="No mathcing query Id.")

        endtime = time.time() + 10  # timeout after 10 seconds
        user_msg, assistant_msg = None, None

        while (user_msg is None or assistant_msg is None) and time.time() < endtime:
            user_entry = next((m for m in reversed(query_msg)
                              if m["role"] == "user"), None)
            if user_entry:
                user_msg = user_entry["content"]
                papers = next(
                    (m["papers"]
                     for m in reversed(query_msg) if m["role"] == "user"),
                    None
                )

                assistant_msg = next(
                    (m["content"]
                     for m in reversed(query_msg) if m["role"] == "assistant"),
                    None
                )

                papers = papers or next(
                    (m["papers"]
                     for m in reversed(query_msg) if m["role"] == "assistant"),
                    None
                )
            if user_msg and assistant_msg:
                break
            time.sleep(0.1)  # avoid busy waiting

        if not user_msg or not assistant_msg:
            raise HTTPException(
                status_code=404, detail="Incomplete conversation.")

        return JSONResponse(content={"user": user_msg, "assistant": assistant_msg, "papers": papers})

    except Exception as e:
        logger.error(f"Error reading chat history for {task_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to read chat history.")


@app.delete("/notebook/{notebook_id}/content/{query_id}")
async def delete_chat_msg(notebook_id: str, query_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    chat_path = os.path.join(config.USERS_ROOT, f"{task_id}.chat.json")
    if not os.path.exists(chat_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")

    try:
        diff = await Notebook_manager.delete_query(task_id, query_id)

        if diff == 0:
            # Nothing was deleted
            raise HTTPException(
                status_code=404, detail="No messages found with given query_id.")

        return JSONResponse(content={"status": "success", "deleted": diff})

    except Exception as e:
        logger.error(f"Error deleting chat message for {task_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to delete chat message.")


@app.patch("/notebook/{notebook_id}/back/{query_id}")
async def delete_chat_msg_from(notebook_id: str, query_id: str, keepUserMsg: bool = False, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    chat_path = os.path.join(config.USERS_ROOT, f"{task_id}.chat.json")
    if not os.path.exists(chat_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")

    try:
        diff = await Notebook_manager.delete_queries_from(
            task_id, query_id, keepUserMsg)

        if diff == 0:
            raise HTTPException(
                status_code=404, detail="No messages found with given query_id.")

        return JSONResponse(content={"status": "success", "deleted": diff})

    except Exception as e:
        logger.error(f"Error deleting chat messages for {task_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to delete chat messages.")


@app.get("/notebook/{notebook_id}/papers/{query_id}")
async def get_chat_papers(notebook_id: str, query_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    chat_path = os.path.join(config.USERS_ROOT, f"{task_id}.chat.json")
    if not os.path.exists(chat_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")

    try:
        history = await Notebook_manager.load_history(task_id)

        endtime = time.time() + 10  # timeout after 10 seconds
        user_msg, papers = None, None

        while user_msg is None and time.time() < endtime:
            user_entry = next((m for m in reversed(history)
                              if m["role"] == "user" and m.get("query_id") == query_id), None)
            if user_entry:
                user_msg = user_entry["content"]
                papers = user_entry["papers"]
                papers = papers or next(
                    (m["papers"] for m in reversed(history)
                     if m["role"] == "assistant" and m.get("query_id") == query_id),
                    None
                )
            if user_msg:
                break
            time.sleep(0.1)  # avoid busy waiting

        if not user_msg:
            raise HTTPException(
                status_code=404, detail="Failed to load papers.")

        return JSONResponse(content={"papers": papers})

    except Exception as e:
        logger.error(f"Error loading papers for {task_id}/{query_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to load papers.")


@app.patch("/notebook/{notebook_id}/scores/{paperId}")
async def set_paper_scores(notebook_id: str, paperId: str, scores: dict, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    try:
        changedId = await Notebook_manager.set_paper_score(task_id, paperIds=paperId, scores=scores)

        return JSONResponse(content={"papers": changedId, "scores": scores})

    except Exception as e:
        logger.error(
            f"Error setting new scores for paper {paperId} in {task_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to set paper scores.")


@app.get("/notebook/{notebook_id}/chat/history")
async def get_history(notebook_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    history = await Notebook_manager.load_history(task_id)
    history = [m for m in history if m.get("access") == 'all']

    return history


@app.get("/notebook/{notebook_id}/chat/result")
async def get_latest_result(notebook_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    result = get_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"No result found for {task_id}")
    return JSONResponse(content={"task_id": task_id, "notebook": result})


@app.patch("/notebook/{notebook_id}/rename/{new_title}")
async def rename_notebook(notebook_id: str, new_title: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    await Notebook_manager.rename(task_id, new_title)

    return JSONResponse(content={"task_id": task_id, "notebook_id": notebook_id, "notebook_title": new_title})


@app.delete("/notebook/{notebook_id}/delete")
async def delete_notebook(notebook_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    task_id = f"{user_id}/{notebook_id}"

    await Notebook_manager.delete(task_id)

    return JSONResponse(content={"task_id": task_id, "notebook_id": notebook_id})


@app.get("/notebook")
async def get_user_notebooks(decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    notebook_list = await Notebook_manager.list(user_id)
    return JSONResponse(content=notebook_list)


@app.get("/notebook/new_task_id")
def generate_task_id(decoded_token: dict = Depends(verify_token)):
    user_id = safe_user_id(decoded_token["uid"])
    print(user_id)
    return JSONResponse(content=create_task_id(user_id))


@app.get("/admin/delete_ghost_users")
async def delete_notebook(decoded_token: dict = Depends(verify_token)):
    if not decoded_token.get("admin"):
        raise HTTPException(status_code=403, detail="Admins only")

    user_ids = list_users()
    firebase_users = get_firebase_users()
    valid_users = set()
    for uid in firebase_users:
        safe_uid = safe_user_id(uid)
        valid_users.add(safe_uid)

    users_to_delete = [uid for uid in user_ids if uid not in {
        v[:len(uid)] for v in valid_users}]

    deleted_users = []
    for uid in users_to_delete:
        res = delete_user(uid)
        deleted_users.append(res.get("user_id"))

    return JSONResponse(content={"deleted_user_ids": deleted_users})


@app.post("/admin/set_admin_rights")
def set_admin_rights(payload: dict, decoded_token: dict = Depends(verify_token)):
    if not decoded_token.get("admin"):
        raise HTTPException(status_code=403, detail="Admins only")

    uid = payload.get("uid")
    set_admin_claim([uid])
    return JSONResponse(content={"message": f"{uid} has now admin access"})


@app.post("/admin/revoke_admin_rights")
def set_admin_rights(payload: dict, decoded_token: dict = Depends(verify_token)):
    if not decoded_token.get("admin"):
        raise HTTPException(status_code=403, detail="Admins only")

    if decoded_token["uid"] == uid:
        raise HTTPException(
            status_code=400, detail="Can't revoke your own admin rights here")

    uid = payload.get("uid")
    revoke_admin_claim([uid])
    return JSONResponse(content={"message": f"Admin rights have been revoked for {uid}"})


@app.get("/auth/validate_admin")
def validate_admin(decoded_token: dict = Depends(verify_token)):
    """
    Called by Nginx (via auth_request) to check if the caller’s token has admin=True.
    Returns 200 if admin; 403 otherwise.
    """
    if not decoded_token.get("admin"):
        raise HTTPException(status_code=403, detail="Admins only")
    return JSONResponse(content={"ok": True})


# Prometheus metrics endpoint: keep it at the bottom
@app.get("/metrics")
def metrics():
    data = generate_latest(_custom_registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("asxai.services.chat.chat_api:app",
                host="0.0.0.0", port=8000, reload=False)
