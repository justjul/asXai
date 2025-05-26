from confluent_kafka.admin import AdminClient, NewTopic
import time
import json
from confluent_kafka import Consumer
import asyncio
import threading
from pathlib import Path
import os
import hashlib

import config
from asxai.vectorDB import OllamaManager
import torch
import numpy as np
import joblib

from typing import Union, List
from asxai.utils import AsyncRunner
from asxai.logger import get_logger
from asxai.utils import load_params, running_inside_docker

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
chat_config = params["chat"]

message_queue = asyncio.Queue()
async_runner = AsyncRunner()


def hash_query(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


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


def kafka_poller(consumer, queue: asyncio.Queue):
    while True:
        msg = consumer.poll(1.0)
        if msg and not msg.error():
            payload = json.loads(msg.value())
            async_runner.run(queue.put(payload))


async def ollama_chat(payload, ollama_manager):
    task_id = payload["task_id"]
    user_id = payload['user_id']
    query_id = payload['query_id']
    notebook_id = payload['notebook_id']
    user_message = payload["content"]
    model_name = payload.get("model", ollama_manager.model_name)
    model_name = ollama_manager.resolve_model(model_name)
    context = load_chat_context(task_id=task_id)
    # if not context:
    # context = load_retrieved_context(task_id=task_id)
    # save_chat_context(task_id, context)

    #     user_message = context[-1]['content']
    #     messages = context
    # else:
    messages = context + [{"role": "user", "content": user_message}]

    try:
        print(messages)
        full_response = ""
        ollama_streamer = await ollama_manager.generate(
            model=model_name,
            messages=messages,
            stream=True,
            options={'temperature': 0.5}
        )
        logger.info(f"Query sent to Ollama for task_id={task_id}")

        buffer = ""
        stream_path = os.path.join(config.USERS_ROOT, f"{task_id}.stream")
        with open(stream_path, "w") as stream_file:
            stream_file.write(f"\n🧑[You]:\n\n {user_message}\n")
            stream_file.write(f"\n🤖[asXai]\n\n")
        with open(stream_path, "a") as stream_file:
            stream_file.write("")
            async for chunk in ollama_streamer:
                token = chunk["message"]["content"]
                buffer += token

                if len(buffer) > 100:
                    stream_file.write(buffer)
                    stream_file.flush()
                    buffer = ""
                full_response += token

            if len(buffer) > 0:
                stream_file.write(buffer)

            if "ASK" in full_response or "QDRANT" in full_response:
                search_qdrant = True
                final_msg = "\n<SEARCHING_QDRANT>\n"
            else:
                search_qdrant = False
                final_msg = "\n<END_OF_MESSAGE>\n"
            stream_file.write(final_msg)
            stream_file.flush()

        new_context = [
            {"role": "user", "content": user_message,
                "task_id": task_id, "timestamp": time.time(),
                "user_id": user_id, "notebook_id": notebook_id,
                "query_id": query_id, "model": model_name},
            {"role": "assistant", "content": full_response,
                "task_id": task_id, "timestamp": time.time(),
                "user_id": user_id, "notebook_id": notebook_id,
                "query_id": query_id, "model": model_name}
        ]
        save_chat_context(task_id, new_context)

        logger.info(f"Streaming complete for task_id={task_id}")

    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
    return full_response


async def worker_loop(ollama_manager):
    while True:
        payload = await message_queue.get()
        asyncio.create_task(ollama_chat(payload, ollama_manager))


def run_chat_worker():
    ollama_manager = OllamaManager()

    KAFKA_BOOTSTRAP = "kafka:9092" if running_inside_docker() else "localhost:29092"
    CHAT_REQ_TOPIC = "notebook-requests"

    create_topic_if_needed(KAFKA_BOOTSTRAP, CHAT_REQ_TOPIC)
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP,
        'group.id': 'chat-group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe([CHAT_REQ_TOPIC])

    async_runner.run(ollama_manager.is_model_pulled())

    threading.Thread(target=kafka_poller,
                     args=(consumer, message_queue),
                     daemon=True).start()
    async_runner.run(worker_loop(ollama_manager))
    logger.info("Chat worker ready")

    # while True:
    #     msg = consumer.poll(1.0)
    #     if msg and not msg.error():
    #         try:
    #             payload = json.loads(msg.value())
    #             full_response = async_runner.run(ollama_chat(payload))
    #         except Exception as e:
    #             logger.error(f"Malformed message or processing error: {e}")


def save_chat_context(task_id: str, new_data: list, append=True):
    users_root = config.USERS_ROOT
    full_path = os.path.join(users_root, f"{task_id}.chat.json")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if append and os.path.exists(full_path):
        try:
            with open(full_path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
        new_data = existing + new_data

    with open(full_path, "w") as f:
        json.dump(new_data, f)


def load_chat_context(task_id: str):
    path = os.path.join(config.USERS_ROOT, f"{task_id}.chat.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load chat context for {task_id}: {e}")
    return []


def load_retrieved_context(task_id: str):
    path = os.path.join(config.USERS_ROOT, f"{task_id}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                res = json.load(f)
                query = res['query']
                query_id = res['query_hash']
                user_id = res['user_id']
                notebook_id = res['notebook_id']
                references = [pl for pl in res['result'] if pl['score'] > 0]
                retrieved_context = {
                    "role": "user",
                    "content": chat_config['synth_instruct'],
                    "task_id": task_id,
                    "timestamp": time.time(),
                    "user_id": user_id,
                    "notebook_id": notebook_id,
                    "query_id": query_id
                }
                formatted_refs = ""
                for i, paper in enumerate(references[:10]):
                    formatted_refs += f"---------\n"
                    formatted_refs += f"- ARTICLE #{i}\n"
                    formatted_refs += f"  Title: {paper['title']}\n"
                    formatted_refs += f"  Authors Names: {paper['authorName']}\n"
                    # formatted_refs += f"  Main text: {paper['main_text']}\n\n"
                    formatted_refs += f"Most relevant chunks: "
                    for chunk in paper['best_chunks']:
                        formatted_refs += f"'{chunk}'\n"
                    formatted_refs += f"\n"

                content_msg = chat_config.get(
                    'synth_instruct', "<REFS>\n\n<QUERY>")
                content_msg = content_msg.replace('<REFS>', formatted_refs)
                content_msg = content_msg.replace('<QUERY>', query)
                retrieved_context['content'] = content_msg

                model_name = 'search-worker'
                retrieved_context = [
                    {"role": "system", "content": "You are a helpful academic assistant. Answer the user's questions using only the provided excerpts of the articles. \
                     If you can't find the answer in the provided excerpts, just return 'ASK QDRANT.",
                        "task_id": task_id, "timestamp": time.time(),
                        "user_id": user_id, "notebook_id": notebook_id,
                        "query_id": query_id, "model": model_name}
                ]
                for i, paper in enumerate(references[:10]):
                    formatted_ref = ""
                    formatted_ref += f"---------\n"
                    formatted_ref += f"- ARTICLE {paper['paperId']}\n"
                    formatted_ref += f"  Title: {paper['title']}\n"
                    formatted_ref += f"  Authors Names: {paper['authorName']}\n"
                    # formatted_refs += f"  Main text: {paper['main_text']}\n\n"
                    formatted_ref += f"Most relevant chunks: "
                    for chunk in paper['best_chunks'][:1]:
                        formatted_ref += f"'{chunk}'\n"
                    formatted_ref += f"\n"
                    retrieved_context += [
                        {"role": "user", "content": formatted_ref,
                         "task_id": task_id, "timestamp": time.time(),
                         "user_id": user_id, "notebook_id": notebook_id,
                         "query_id": query_id, "model": model_name}
                    ]

                return retrieved_context
        except Exception as e:
            logger.warning(
                f"Failed to load retrieved context for {task_id}: {e}")
    return []


def start_chat_worker():
    t = threading.Thread(target=run_chat_worker, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    async_runner.run(run_chat_worker())
    # t = threading.Thread(target=run_chat_worker, daemon=True)
    # t.start()
    # t.join()
