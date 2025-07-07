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
from asxai.llms import InferenceManager
from .notebook_manager import NotebookManager
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
search_config = params["search"]

message_queue = asyncio.Queue()
cancel_queue = asyncio.Queue()
async_runner = AsyncRunner()
SEARCH_HOST = "search-api" if running_inside_docker() else 'localhost'
SEARCH_PORT = 8000 if running_inside_docker() else 8100

Notebook_manager = NotebookManager(config.USERS_ROOT)


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


async def stream_to_queue(streamer, queue):
    async for chunk in streamer:
        await queue.put(chunk)
    await queue.put(None)


async def response_to_queue(coro, queue):
    try:
        draft = await coro
        await queue.put(draft)
    except Exception as e:
        await queue.put(e)


class ChatManager:
    def __init__(self, payload, inference_manager):
        self.task_id = payload["task_id"]
        self.user_id = payload['user_id']
        self.query_id = payload['query_id']
        self.notebook_id = payload['notebook_id']
        self.user_message = payload["content"]
        self.model_name = payload.get("model", inference_manager.model_name)
        self.topK = payload["topK"]
        self.paperLock = payload["paperLock"]
        self.mode = payload["mode"]
        self.chat_path = os.path.join(
            config.USERS_ROOT, f"{self.task_id}.chat.json")
        self.search_path = os.path.join(
            config.USERS_ROOT, f"{self.task_id}.json")
        self.summaries_path = os.path.join(
            config.USERS_ROOT, f"{self.task_id}.summaries.json")
        self.stream_path = os.path.join(
            config.USERS_ROOT, f"{self.task_id}.stream")
        with open(self.stream_path, "w") as stream_file:
            stream_file.write("\n")
        os.makedirs(os.path.dirname(self.chat_path), exist_ok=True)

        if self.mode in ['regenerate']:
            self.delete_assistant_msg(mode=self.mode)
            self.user_message = self.get_user_message()
            self.delete_user_msg()

        ts = self.get_timestamp()
        self.timestamp = ts or time.time()
        self.timestamp_cutoff = ts or 2 * time.time()
        if ts and self.mode in ['reply']:
            self.delete_assistant_msg(['reply', 'expand'])
            self.delete_user_msg()
            logger.info(
                f"{self.mode} mode on previous msg for {self.task_id}/{self.query_id}")
        if ts and self.mode in ['expand']:
            self.delete_assistant_msg(['expand'])
            logger.info(
                f"{self.mode} mode on previous msg for {self.task_id}/{self.query_id}")

    def get_timestamp(self):
        if os.path.exists(self.chat_path):
            try:
                with open(self.chat_path) as f:
                    history = json.load(f)
                    ts = next((m.get("timestamp") for m in reversed(history) if m.get(
                        "query_id") == self.query_id), None)
                return ts
            except Exception as e:
                logger.warning(
                    f"Failed to estimate timestamp cutoff for {self.task_id}: {e}")
        return None

    def get_user_message(self):
        if os.path.exists(self.chat_path):
            try:
                with open(self.chat_path) as f:
                    history = json.load(f)
                    content = next((m.get("content") for m in reversed(history) if m.get(
                        "query_id") == self.query_id and m.get("role") == 'user'), None)
                return content
            except Exception as e:
                logger.warning(
                    f"Failed to fetch user's message from chat history for {self.task_id}: {e}")
        return None

    def delete_assistant_msg(self, mode: List[str]):
        if os.path.exists(self.chat_path):
            try:
                with open(self.chat_path, "r") as f:
                    history = json.load(f)
                    history = [m for m in history
                               if m.get("query_id") != self.query_id
                               or m.get("role") != 'assistant'
                               or m.get("mode") not in mode]
                with open(self.chat_path, "w") as f:
                    json.dump(history, f)
            except Exception as e:
                logger.warning(
                    f"Failed to delete asistant message from chat context for {self.task_id}: {e}")
        return []

    def delete_user_msg(self):
        if os.path.exists(self.chat_path):
            try:
                with open(self.chat_path, "r") as f:
                    history = json.load(f)
                    history = [m for m in history if m.get(
                        "query_id") != self.query_id or m.get("role") != 'user']
                with open(self.chat_path, "w") as f:
                    json.dump(history, f)
            except Exception as e:
                logger.warning(
                    f"Failed to delete user's message from chat context for {self.task_id}: {e}")
        return []

    def load_chat_history(self):
        if os.path.exists(self.chat_path):
            try:
                with open(self.chat_path) as f:
                    history = json.load(f)
                    history = [m for m in history if m.get(
                        "timestamp") <= self.timestamp_cutoff]
                return history
            except Exception as e:
                logger.warning(
                    f"Failed to load chat context for {self.task_id}: {e}")
        return []

    def fetch_user_msg(self, history: List[dict]):
        user_msg = next((m for m in reversed(history) if m.get(
            'role') == "user" and m.get('query_id') == self.query_id), [])
        return user_msg

    def save_chat_msg(
            self, role, content,
            model, papers, access,
            mode,
            think=None,
            notebook_title=None,
            search_query=None,
            append=True
    ):
        base_msg = {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "notebook_id": self.notebook_id,
            "query_id": self.query_id
        }
        new_msg = {**base_msg,
                   "role": role,
                   "content": content,
                   "think": think,
                   "model": model,
                   "mode": mode,
                   "papers": papers,
                   "access":  access,
                   "search_query": search_query,
                   "notebook_title": notebook_title
                   }

        before, after = [], []
        if append and os.path.exists(self.chat_path):
            try:
                with open(self.chat_path, "r") as f:
                    history = json.load(f)
                    before = [m for m in history if m.get(
                        "timestamp") <= self.timestamp_cutoff]
                    after = [m for m in history if m.get(
                        "timestamp") > self.timestamp_cutoff]
            except Exception as e:
                before, after = [], []
                logger.warning(
                    f"Failed to save chat messages for {self.task_id}: {e}")

        new_data = before + [new_msg] + after

        with open(self.chat_path, "w") as f:
            json.dump(new_data, f)

    def load_chat_summaries(self):
        if os.path.exists(self.summaries_path):
            try:
                with open(self.summaries_path) as f:
                    summaries = json.load(f)
                    summaries = [m for m in summaries if m.get(
                        "timestamp") <= self.timestamp_cutoff]
                    return summaries
            except Exception as e:
                logger.warning(
                    f"Failed to load chat summaries for {self.task_id}: {e}")
        return []

    def save_chat_summary(self, content: str, timestamp: float, query_id: str, append=True):
        base_data = {
            "role": "assistant",
            "task_id": self.task_id,
            "user_id": self.user_id,
            "notebook_id": self.notebook_id,
            "model": "summarizer",
            "access": "assistant",
            "mode": "summary",
            "query_id": self.query_id,
            "papers": None,
            "notebook_title": self.notebook_title
        }
        new_data = {**base_data,
                    "content": content,
                    "timestamp": timestamp,
                    "query_id": query_id,
                    }

        os.makedirs(os.path.dirname(self.summaries_path), exist_ok=True)

        if append and os.path.exists(self.summaries_path):
            try:
                with open(self.summaries_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        new_data = existing + new_data

        with open(self.summaries_path, "w") as f:
            json.dump(new_data, f)

    async def update_summaries(self, history, summaries):
        if self.mode in ["expand", "regenerate"]:
            return []
        if summaries:
            last_summary_date = summaries[-1].get("timestamp", 0)
            recent_history = [
                turns for turns in history if turns["timestamp"] > last_summary_date]
        else:
            recent_history = history

        # Updating chat summaries if needed
        assist_words = [len(turn['content'].split(
            ' ')) for turn in recent_history if turn['role'] == 'assistant' and turn['role'] != 'summarizer']
        assist_words = list(reversed(assist_words))
        assist_cumsum = np.cumsum(assist_words).tolist() or [0]
        print(f"NUMBER OF ASSISTANT WORDS SO FAR: {assist_cumsum[-1]}")
        if assist_cumsum[-1] > 2*chat_config['min_history_length']:
            turn_idx_end = next((i for i, n in enumerate(
                assist_cumsum) if n > chat_config['min_history_length']), None)
            if turn_idx_end is not None:
                history_to_summarize = recent_history[:-turn_idx_end]
                summary = await self.inference_manager.chatSummarize(chat_history=summaries + history_to_summarize)
                summary_ts = history_to_summarize[-1]["timestamp"]
                summary_id = history_to_summarize[-1]["query_id"]
                self.save_chat_summary(
                    content=summary,
                    timestamp=summary_ts,
                    query_id=summary_id
                )
        return summary_ts

    def stream(self, msg):
        with open(self.stream_path, "a") as stream_file:
            stream_file.write(msg)
            stream_file.flush()

    def delete_stream(self):
        if os.path.isfile(self.stream_path):
            os.remove(self.stream_path)

    def submit_search(self, prefix_id: int, query: str, topK: int, paperLock: bool):
        search_query_id = str(prefix_id).strip('_') + '_' + self.query_id
        SEARCH_API_URL = f"http://{SEARCH_HOST}:{SEARCH_PORT}"
        payload = {
            "user_id": self.user_id,
            "notebook_id": self.notebook_id,
            "query_id": search_query_id,
            "query": query,
            "topK": topK,
            "paperLock": paperLock,
        }
        res = requests.post(f"{SEARCH_API_URL}/search", json=payload)

    def load_search_result(self, prefix_id: int):
        search_query_id = str(prefix_id).strip('_') + '_' + self.query_id
        SEARCH_API_URL = f"http://{SEARCH_HOST}:{SEARCH_PORT}"
        timeout = search_config['timeout']
        endtime = time.time() + timeout
        try:
            while True:
                try:
                    res = requests.get(
                        f"{SEARCH_API_URL}/search/{self.task_id}/{search_query_id}").json()
                    res = res['notebook']
                    if (res and search_query_id in {r['query_id'] for r in res}):
                        break
                except Exception as e:
                    logger.warning(
                        f"There was an issue trying to load search results: {e}")
                if time.time() > endtime:
                    break

            papers = []
            match_generator = (
                pl for pl in res if pl['query_id'] == search_query_id and pl.get('paperId'))
            while True:
                try:
                    papers.append(next(match_generator))
                except StopIteration:
                    break

            for paper in papers:
                if paper['openAccessPdf'].startswith("gs://"):
                    arxiv_id = paper['openAccessPdf'].rsplit("/", 1)[-1][:-4]
                    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                    paper['openAccessPdf'] = f"https://arxiv.org/pdf/{arxiv_id}"

        except Exception as e:
            logger.error(
                f"Unable to load search result for {self.task_id}: {e}")

        return papers


async def chat_process(payload, inference_manager):
    try:
        task_id = payload["task_id"]
        search_query = payload["search_query"]
        model_name = payload.get("model", inference_manager.model_name)
        topK = payload["topK"]
        paperLock = payload["paperLock"]

        if payload["content"].lower() == "*notebook update*":
            logger.info("UPDATE MODE")
            payload["mode"] = "update"

        chat_manager = ChatManager(payload, inference_manager)

        # Load existing context from disk (if any).
        history = chat_manager.load_chat_history()
        summaries = chat_manager.load_chat_summaries()

        if summaries:
            last_summary_date = summaries[-1].get("timestamp", 0)
            recent_history = [
                turns for turns in history if turns["timestamp"] > last_summary_date]
            context = summaries + recent_history
        else:
            context = history
            recent_history = history

        # Updating chat summaries if needed
        async_runner.submit(chat_manager.update_summaries(history, summaries))

        if not context:
            title_response = await inference_manager.generateTitle(
                query=chat_manager.user_message,
                model=model_name
            )
            notebook_title = title_response['title']
        else:
            notebook_title = context[-1].get('notebook_title',
                                             chat_manager.user_message)

        full_query = chat_manager.user_message + ' \n'

        if chat_manager.mode in ["reply"]:
            if not search_query:
                startime = time.time()
                chat_manager.stream(" *Processing query* ")
                search_query = await inference_manager.processQuery(
                    query=chat_manager.user_message,
                    chat_history=context,
                    model=model_name
                )
                chat_manager.stream(f"*{round(time.time() - startime)}s*")
            search_query['paperIds'] = []
            search_query['topK'] = topK
            papers = []
        elif chat_manager.mode in ["regenerate", "expand"]:
            user_msg = chat_manager.fetch_user_msg(history)
            search_query = user_msg['search_query']
            papers = user_msg['papers']
            full_query += ' \n'.join(search_query.get(
                'queries', ''))
        elif chat_manager.mode in ["update"]:
            user_msgs = [m for m in history if m.get(
                'role') == "user" and m.get('search_query')]
            search_query = [m['search_query'] for m in user_msgs]
            papers = []

        if search_query and chat_manager.mode in ["reply", "update", "regenerate"]:
            search_queries = [search_query] if not isinstance(
                search_query, list) else search_query

            startime = time.time()
            chat_manager.stream(" *Searching* ")
            for k, search_query in enumerate(search_queries):
                if (not search_query.get('queries', '')
                    or not search_query.get('ethical', True)
                        or not search_query.get('scientific', True)):
                    continue

                chat_manager.submit_search(
                    prefix_id=k,
                    query=search_query,
                    topK=search_query['topK'], paperLock=paperLock
                )

                paper_results = chat_manager.load_search_result(
                    prefix_id=k)
                paper_results = [
                    p for p in paper_results if p['paperId'] not in search_query.get('paperIds', [])[2:]]

                filt_papers = await inference_manager.filterArticles(
                    query='/n'.join(search_query.get('queries', '')),
                    documents=serialize_documents(paper_results),
                    model=model_name
                )
                filt_paperIds = filt_papers.get('paperIds', []) or []
                logger.info(f"filtered paperIds {filt_paperIds}")

                # filtering results based on MCP for article filtering
                paper_results = [
                    p for p in paper_results if p['paperId'] in filt_paperIds]

                search_query['paperIds'].extend(
                    [p['paperId'] for p in paper_results])
                papers.extend(paper_results)

                if not search_query.get('cite_only', False):
                    full_query += ' \n'.join(
                        search_query.get('queries', '')) + ' \n'

            chat_manager.stream(f"*{round(time.time() - startime)}s*")

            if not chat_manager.mode in ["update"]:
                search_query = search_queries[0]
            else:
                search_query = {}

            for paper in papers:
                chat_manager.save_chat_msg(
                    role="user", content=serialize_documents(paper)[0],
                    model="search-worker", papers=None, mode=chat_manager.mode,
                    access="assistant", notebook_title=notebook_title
                )

            chat_manager.save_chat_msg(
                role="user", content=chat_manager.user_message,
                model=model_name, papers=papers, mode=chat_manager.mode,
                access="all", notebook_title=notebook_title,
                search_query=search_query,
            )

        if chat_manager.mode in ["expand"] and search_query and papers:
            startime = time.time()
            chat_manager.stream(" *Generating Plan*")
            generationPlan = await inference_manager.generatePlan(
                query=full_query,
                documents=serialize_documents(papers),
                model=model_name
            )
            chat_manager.stream(f"*{round(time.time() - startime)}s*")
            sections = generationPlan.get('sections')
            abstract = generationPlan.get('abstract')
            instruct_writer = chat_config['instruct_gensection']

            if abstract:
                sections.insert(
                    0,
                    {"title": "abstract",
                     "content": abstract,
                     "paperIds": [p.get("paperId") for p in papers]
                     }
                )
        else:
            paperIds = [p.get("paperId") for p in papers]
            sections = [
                {"title": "", "content": full_query, "paperIds": paperIds}]
            abstract = []
            if not chat_manager.mode in ["update"]:
                if search_query.get('cite_only', False):
                    instruct_writer = chat_config['instruct_insertCitation']
                else:
                    instruct_writer = chat_config['instruct_quickreply']
            else:
                instruct_writer = chat_config['instruct_update']
                if not papers:
                    sections = []

        logger.info(sections)
        for section in sections:
            section_papers = [p for p in papers if p['paperId']
                              in section.get('paperIds', [])]
            section['documents'] = serialize_documents(
                section_papers,
                include_abstract=chat_manager.mode in ["update"]
            )

        startime = time.time()
        chat_manager.stream("\n<WRITING>\n")

        full_response = ""
        think_response = ""
        clear_stream = False
        if abstract:
            chat_manager.stream(f"\n\n{abstract}\n\n")
            clear_stream = True

        queues = [asyncio.Queue() for _ in sections]
        task_writers = []
        for section, queue in zip(sections, queues):
            writer_streamer = await inference_manager.writer(
                title=section['title'],
                content=section.get('content', ''),
                documents=section['documents'],
                instruct=instruct_writer,
                stream=True,  # False,  # True,
                model=model_name,
                top_p=0.1,
            )
            task_writers.append(asyncio.create_task(
                inference_manager.stream_to_queue(writer_streamer, queue)))

        task_editors = []
        full_response = ""
        for section, queue in zip(sections, queues):
            chat_manager.stream(f"\n\n## {section['title']}\n\n")
            section['content'] = ""
            think_response += " \n\n"

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                section['content'] += chunk['content']
                think_response += chunk['think']

                if clear_stream:
                    chat_manager.stream("\n<CLEAR>\n")
                    clear_stream = False

                chat_manager.stream(chunk['content'])

            if (not chat_manager.mode in ["update"] and papers
                    and (not search_query.get('cite_only', False)
                         or chat_manager.mode in ["expand"])):
                content_editor = inference_manager.writer(
                    title=section['title'],
                    content=section.get('content', ''),
                    documents=section['documents'],
                    instruct=chat_config['instruct_contentEditor'],
                    stream=False,  # False,  # True,
                    model=model_name,
                    top_p=0.1,
                )
                task_editors.append(asyncio.create_task(
                    inference_manager.response_to_queue(content_editor, queue)))
            else:
                full_response += section.get('content', '')

        if task_editors:
            chat_manager.stream("\n<EDITING>\n")

            for section, queue in zip(sections, queues):
                edited_response = await queue.get()
                section['content'] = edited_response['content']

                full_response = ""
                for s in sections:
                    if s['title'] != "abstract":
                        full_response += f"\n\n## {s['title']}\n\n"
                    full_response += s['content']
                chat_manager.stream("\n<CLEAR>\n")
                chat_manager.stream(full_response)

            chat_manager.stream("\n<REVISING>\n")

            if not chat_manager.mode in ["update"]:
                context = [msg for msg in context if msg['model']
                           != 'search-worker']
                context += [{'role': 'user', 'content': full_query}]
                final_version = await inference_manager.writer(
                    content=full_response,
                    context=context,
                    instruct=chat_config['instruct_styleEditor'],
                    stream=False,  # False,  # True,
                    model=model_name,
                    top_p=0.1,
                )
                full_response = final_version['content']
                chat_manager.stream("\n<CLEAR>\n")
                chat_manager.stream(full_response)

            await asyncio.gather(*task_editors)

        await asyncio.gather(*task_writers)

        if not think_response.strip("\n").strip(" "):
            think_response = ""

        if full_response:
            chat_manager.save_chat_msg(
                role="assistant", content=full_response, think=think_response,
                model=model_name, mode=chat_manager.mode,
                papers=None, access="all", notebook_title=notebook_title,
            )

        chat_manager.stream("\n<END_OF_MESSAGE>\n")
        logger.info(f"Streaming complete for task_id={task_id}")

        await Notebook_manager.search_cleanup(task_id)

    except Exception as e:
        logger.error(f"Error handling chat request: {e}")

    finally:
        chat_manager.delete_stream()
        await Notebook_manager.chat_cleanup(task_id)
        if chat_manager.mode in ["update"]:
            await Notebook_manager.set_update_time(task_id)
        deleted_notebooks = await Notebook_manager.notebooks_cleanup(chat_manager.user_id)

    return full_response


task_registry = {}


async def worker_loop(inference_manager):
    while True:
        payload = await message_queue.get()
        task_id = payload['task_id']

        # if a task is already running for that notebook
        # we recycle it until it's done
        if task_id in task_registry:
            logger.info("Notebook processing a previous query")
            await asyncio.sleep(0.2)
            await message_queue.put(payload)
            logger.info("Latest query recycled to waiting queue")
            continue

        task = asyncio.create_task(chat_process(payload, inference_manager))
        task_registry[task_id] = task
        task.add_done_callback(lambda t: task_registry.pop(task_id, None))


async def cancel_loop():
    while True:
        payload = await cancel_queue.get()
        task_id = payload['task_id']
        task = task_registry.get(task_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancelled task {task_id}")
        else:
            logger.warning(f"Task {task_id} not found or already finished")


async def chat_loops(inference_manager):
    logger.info("Chat worker ready")
    await asyncio.gather(
        worker_loop(inference_manager),
        cancel_loop()
    )


def run_chat_worker():
    inference_manager = InferenceManager()

    KAFKA_BOOTSTRAP = "kafka:9092" if running_inside_docker() else "localhost:29092"
    CHAT_REQ_TOPIC = "notebook-requests"
    CHAT_CANCEL_TOPIC = "notebook-cancel"

    create_topic_if_needed(KAFKA_BOOTSTRAP, CHAT_REQ_TOPIC)
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP,
        'group.id': 'chat-group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe([CHAT_REQ_TOPIC])

    create_topic_if_needed(KAFKA_BOOTSTRAP, CHAT_CANCEL_TOPIC)
    consumer_cancel = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP,
        'group.id': 'chat-cancel-group',
        'auto.offset.reset': 'earliest'
    })
    consumer_cancel.subscribe([CHAT_CANCEL_TOPIC])

    async_runner.run(inference_manager.is_model_pulled())

    threading.Thread(target=kafka_poller,
                     args=(consumer, message_queue),
                     daemon=True).start()

    threading.Thread(target=kafka_poller,
                     args=(consumer_cancel, cancel_queue),
                     daemon=True).start()

    async_runner.run(chat_loops(inference_manager))


def serialize_documents(documents: List[dict], include_abstract: bool = False):
    if not isinstance(documents, list):
        documents = [documents]

    if not documents:
        return ["No article relevant to the query were found in the database"]

    serialized = []
    try:
        for paper in documents:
            formatted_ref = ""
            formatted_ref += f"--- Article ---\n"
            formatted_ref += f"Article ID: [{paper['paperId']}]\n"
            formatted_ref += f"Title: {paper['title']}\n"
            formatted_ref += f"Authors: {paper['authorName'].split(',', 1)[0]} et al.\n"
            formatted_ref += f"Date: {paper['publicationDate']}\n"
            # formatted_refs += f"  Main text: {paper['main_text']}\n\n"
            if include_abstract:
                formatted_ref += f"Abstract:\n"
                formatted_ref += f"{paper['abstract']}\n"
            formatted_ref += f"Excerpt:\n"
            for chunk in paper['best_chunks'][:1]:
                formatted_ref += f"'{chunk}'\n"
            formatted_ref += "\n"

            serialized.append(formatted_ref)
    except Exception as e:
        logger.error(f"Unable to serialize articles: {e}")

    return serialized


def start_chat_worker():
    t = threading.Thread(target=run_chat_worker, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    async_runner.run(run_chat_worker())
    # t = threading.Thread(target=run_chat_worker, daemon=True)
    # t.start()
    # t.join()
