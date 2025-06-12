from confluent_kafka.admin import AdminClient, NewTopic
import time
import json
from confluent_kafka import Consumer
import asyncio
import threading
from pathlib import Path
import os
import hashlib
import requests

import config
from asxai.llms import OllamaManager
import torch
import numpy as np
import re
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
SEARCH_HOST = "search-api" if running_inside_docker() else 'localhost'
SEARCH_PORT = 8000 if running_inside_docker() else 8100


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
    topK = payload["topK"]
    paperLock = payload["paperLock"]
    model_name = ollama_manager.resolve_model(model_name)

    parse_instruct = chat_config["instruct_parse"]
    parsed_user_message = await ollama_manager.parse(user_message, parse_instruct=parse_instruct)
    user_message_cleaned = parsed_user_message.get(
        'cleaned_query') or parsed_user_message.get('query') or None
    payload_filters = parsed_user_message
    print(payload_filters)

    # Load existing context from disk (if any).
    context = load_chat_context(task_id=task_id)

    instruct_init = chat_config["instruct_init"]
    if not context:
        prompt = (
            f"Summarize the following user request in exactly three words. "
            f"Do not return anything else:\n\n\"{user_message_cleaned}\""
        )
        notebook_title = await ollama_manager.generate(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.0},  # deterministic
        )
    else:
        notebook_title = context[-1].get('notebook_title', user_message)

    messages = context + \
        [{"role": "user", "content": user_message_cleaned + instruct_init}]

    try:
        done = False
        keep_streaming = False
        search_context = []
        papers = None
        full_response = ""
        while not done:
            ollama_streamer = await ollama_manager.generate(
                model=model_name,
                messages=messages,
                stream=True,
                options={'temperature': 0.5}
            )
            logger.info(f"Query sent to Ollama for task_id={task_id}")

            buffer = ""
            stream_path = os.path.join(config.USERS_ROOT, f"{task_id}.stream")
            if not keep_streaming:
                with open(stream_path, "w") as stream_file:
                    stream_file.write(f"\n")
                    stream_file.flush()
            with open(stream_path, "a") as stream_file:
                stream_file.write("")
                stream_file.flush()
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

                if not done and "SEARCHING" in full_response:
                    done = False
                    final_msg = "\n<SEARCHING>\n"
                    response_parsed = full_response.split("SEARCHING:")
                    if len(response_parsed) > 1:
                        search_query = ' '.join(response_parsed[1:]).strip()
                        full_response = response_parsed[0].strip()
                    else:
                        full_response = ""
                else:
                    done = True
                    final_msg = "\n<END_OF_MESSAGE>\n"
                stream_file.write(final_msg)
                stream_file.flush()

            keep_streaming = True

            if done == False:
                if search_query:
                    print({'query': search_query,
                           **payload_filters, })
                    submit_search(user_id=user_id,
                                  notebook_id=notebook_id,
                                  query_id=query_id,
                                  query={'query': search_query,
                                         **payload_filters, },
                                  topK=topK,
                                  paperLock=paperLock)
                    search_context, papers = load_search_result(task_id=task_id,
                                                                user_id=user_id,
                                                                notebook_id=notebook_id,
                                                                query_id=query_id,
                                                                topK=topK)

                instruct_refine = chat_config["instruct_refine"]
                messages = context + search_context + \
                    [{"role": "user", "content": user_message + instruct_refine}]

        new_context = search_context
        new_context.extend([
            {"role": "user", "content": user_message,
                "task_id": task_id, "timestamp": time.time(),
                "user_id": user_id, "notebook_id": notebook_id,
                "query_id": query_id, "model": model_name,
                "papers": None, "access":  "all",
                'notebook_title': notebook_title},
            {"role": "assistant", "content": full_response,
                "task_id": task_id, "timestamp": time.time(),
                "user_id": user_id, "notebook_id": notebook_id,
                "query_id": query_id, "model": model_name,
                "papers": papers, "access":  "all",
                'notebook_title': notebook_title}
        ])
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


def submit_search(user_id: str,
                  notebook_id: str,
                  query_id: str,
                  query: str,
                  topK: int,
                  paperLock: bool):
    SEARCH_API_URL = f"http://{SEARCH_HOST}:{SEARCH_PORT}"
    payload = {
        "user_id": user_id,
        "notebook_id": notebook_id,
        "query_id": query_id,
        "query": query,
        "topK": topK,
        "paperLock": paperLock,
    }
    res = requests.post(f"{SEARCH_API_URL}/search", json=payload)


def load_search_result(user_id: str,
                       notebook_id: str,
                       task_id: str,
                       query_id: str,
                       topK: int):
    SEARCH_API_URL = f"http://{SEARCH_HOST}:{SEARCH_PORT}"
    timeout = 15
    endtime = time.time() + timeout
    try:
        while True:
            try:
                res = requests.get(f"{SEARCH_API_URL}/search/{task_id}").json()
                res = res['notebook']
                if (res and query_id in {r['query_id'] for r in res[-5:]}):
                    break
            except Exception as e:
                logger.warning(
                    f"There was an issue trying to load search results: {e}")
            if time.time() > endtime:
                break

        results = []
        match_generator = (pl for pl in res if pl['query_id'] == query_id)
        while len(results) < topK:
            try:
                results.append(next(match_generator))
            except StopIteration:
                break

        model_name = 'search-worker'
        search_context = []
        papers = []
        for paper in results:
            if paper['openAccessPdf'].startswith("gs://"):
                arxiv_id = paper['openAccessPdf'].rsplit("/", 1)[-1][:-4]
                arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                paper['openAccessPdf'] = f"https://arxiv.org/pdf/{arxiv_id}"
            papers.append(paper)
            formatted_ref = ""
            formatted_ref += f"---------\n"
            formatted_ref += f"-  ARTICLE_ID: {paper['paperId']}\n"
            formatted_ref += f"-  PDF URL: {paper['openAccessPdf']}\n"
            formatted_ref += f"-  TITLE: {paper['title']}\n"
            formatted_ref += f"-  AUTHORS: {paper['authorName']}\n"
            formatted_ref += f"-  PUBLICATION DATE: {paper['publicationDate']}\n"
            # formatted_refs += f"  Main text: {paper['main_text']}\n\n"
            formatted_ref += f"-  EXCERPT: "
            for chunk in paper['best_chunks'][:1]:
                formatted_ref += f"'{chunk}'\n"
            formatted_ref += f"\n"

            search_context += [{"role": "user", "content": formatted_ref,
                                "task_id": task_id, "timestamp": time.time(),
                                "user_id": user_id, "notebook_id": notebook_id,
                                "query_id": query_id, "model": model_name,
                                "access":  "assistant"}
                               ]
    except Exception as e:
        logger.error(f"Unable to load search result for {task_id}: {e}")

    return search_context, papers


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
                    formatted_ref += f"- ARTICLE_ID {paper['paperId']}\n"
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
