# Notebook API – collects chat prompts, notebook results and send back Ollama answers

from confluent_kafka.admin import AdminClient, NewTopic
import json
from uuid import uuid4
from typing import Optional, List
import os
import time

from fastapi import FastAPI,  HTTPException
import uvicorn
import hashlib
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from confluent_kafka import Producer, Consumer

import config
from asxai.logger import get_logger
from asxai.utils import load_params, running_inside_docker

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
chat_config = params["chat"]
search_config = params["search"]

KAFKA_BOOTSTRAP = "kafka:9092" if running_inside_docker() else "localhost:29092"
CHAT_REQ_TOPIC = "notebook-requests"
HASH_LEN = search_config['hash_len']

USERS_ROOT = config.USERS_ROOT


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


class ChatRequest(BaseModel):
    message: str
    user_id: str
    notebook_id: str
    model: str = "default"


@app.post("/notebook/{task_id:path}/chat")
async def submit_chat(task_id: str, req: ChatRequest):
    payload = {
        "task_id": task_id,
        "query_id": hash_id(req.message),
        "user_id": req.user_id,
        "notebook_id": req.notebook_id,
        "content": req.message,
        "role": "user",
        "model": req.model,
        "timestamp": time.time(),
        "done": 0
    }
    print(req.model)
    producer.produce(CHAT_REQ_TOPIC, key=task_id, value=json.dumps(payload))
    producer.flush()
    return {"status": "submitted", "task_id": task_id}


@app.get("/notebook/{task_id:path}/stream")
def stream_response(task_id: str):
    def event_stream():
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
                    yield f"data: {line}\n\n"

                    if line == "<END_OF_MESSAGE>":
                        break
        finally:
            # Cleanup the stream file after completion
            try:
                os.remove(stream_path)
            except Exception as e:
                logger.warning(
                    f"Could not delete stream file {stream_path}: {e}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/notebook/{task_id:path}/chat/final")
def get_final_chat(task_id: str):
    chat_path = os.path.join(config.USERS_ROOT, f"{task_id}.chat.json")
    if not os.path.exists(chat_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")

    try:
        with open(chat_path) as f:
            history = json.load(f)

        endtime = time.time() + 10  # timeout after 10 seconds
        user_msg, assistant_msg = None, None
        query_id = None

        while (user_msg is None or assistant_msg is None) and time.time() < endtime:
            user_entry = next((m for m in reversed(history)
                              if m["role"] == "user"), None)
            if user_entry:
                user_msg = user_entry["content"]
                query_id = user_entry.get("query_id")
            if query_id:
                assistant_msg = next(
                    (m["content"] for m in reversed(history)
                     if m["role"] == "assistant" and m.get("query_id") == query_id),
                    None
                )
            if user_msg and assistant_msg:
                break
            time.sleep(0.1)  # avoid busy waiting

        if not user_msg or not assistant_msg:
            raise HTTPException(
                status_code=404, detail="Incomplete conversation.")

        return {
            "user": user_msg,
            "assistant": assistant_msg
        }

    except Exception as e:
        logger.error(f"Error reading chat history for {task_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to read chat history.")


@app.get("/notebook/{task_id:path}/history")
def get_history(task_id: str):
    chat_path = os.path.join(config.USERS_ROOT, f"{task_id}.chat.json")
    if os.path.exists(chat_path):
        with open(chat_path) as f:
            return json.load(f)
    return {"task_id": task_id, "messages": []}


@app.get("/notebook/{task_id:path}/result")
async def get_latest_result(task_id: str):
    result = get_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"No result found for {task_id}")
    return {"task_id": task_id, "notebook": result}


@app.get("/notebook/task_id")
def generate_task_id(user_id: str, notebook_id: str):
    return {"task_id": make_task_id(user_id, notebook_id)}


if __name__ == "__main__":
    uvicorn.run("asxai.services.chat.api:app",
                host="0.0.0.0", port=8000, reload=False)
