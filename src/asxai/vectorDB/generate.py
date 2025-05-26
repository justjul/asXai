import os
import config
import asyncio

from ollama import AsyncClient
import requests

from typing import List, Tuple, Union, TypedDict, AsyncGenerator
from pathlib import Path
import json
import uuid
import time
from dateutil.parser import parse
from datetime import datetime

from tqdm import tqdm
from asxai.utils import load_params
from asxai.utils import running_inside_docker

from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
embedding_config = params["embedding"]
qdrant_config = params["qdrant"]
ollama_config = params["ollama"]


def extract_parsed_fields(text: str):
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        json_block = text[start:end]
        print(json_block)
        json_block = json_block.replace("None", "null")
        dic = json.loads(json_block)
        print(dic)
        default = datetime(1, 1, 1)  # defaults everything to Jan 1

        def safe(val):
            try:
                parse(val, default=default).strftime("%Y-%m-%d")
                return True
            except Exception:
                return False

        norm_dic = {'query': next((val for key, val in dic.items() if 'query' in key and val != 'null'), None),
                    'authorName': next((val for key, val in dic.items() if 'name' in key and val != 'null'), None),
                    'publicationDate_start': next(
                        (parse(val, default=default).strftime("%Y-%m-%d")
                         for key, val in dic.items() if 'start_date' in key and safe(val)), None),
                    'publicationDate_end': next(
                        (parse(val, default=default).strftime("%Y-%m-%d")
                         for key, val in dic.items() if 'end_date' in key and safe(val)), None)}
        return norm_dic
    except (ValueError, json.JSONDecodeError):
        return {}


class OllamaManager:
    def __init__(self,
                 model_name: str = ollama_config["model_name"],
                 docker_timeout: float = ollama_config["docker_timeout"],
                 watch_tmp_timeout: float = ollama_config["watch_tmp_timeout"],
                 max_threads: int = ollama_config["max_threads"]):

        self.container_name = "ollama"
        self.model_name = model_name

        self.port = ollama_config["port"]
        self.host = "ollama" if running_inside_docker(
        ) else ollama_config["host"]

        self.meta_fields = {'paperId', 'fieldsOfStudy', 'venue', 'authorName',
                            'citationCount', 'influentialCitationCount',
                            'publicationDate', 'publicationYear'}

        self.docker_timeout = docker_timeout
        self.watch_tmp_timeout = watch_tmp_timeout
        self.max_threads = max_threads

        self.is_ollama_running()
        self.client = AsyncClient(host=f"http://{self.host}:{self.port}",
                                  timeout=120.0)
        self.semaphore = asyncio.Semaphore(self.max_threads)

    def is_ollama_running(self):
        ready = False
        endtime = time.time() + self.docker_timeout
        url = f"http://{self.host}:{self.port}"
        n_retry = 0

        while time.time() < endtime:
            try:
                response = requests.get(url, timeout=2.0)
                if response.status_code == 200:
                    if n_retry > 0:
                        logger.info("Ollama service is up and running.")
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            logger.debug("Ollama not ready yet. Retrying in 3 seconds...")
            n_retry += 1
            time.sleep(3)

        if not ready:
            logger.error(
                f"Ollama container '{self.container_name}' did not become ready within timeout.")

    async def is_model_pulled(self):
        try:
            model_list = await self.client.list()
            models = model_list.models
            available_models = [model.model for model in models]

            if self.model_name not in available_models:
                logger.info(f"Pulling model '{self.model_name}'...")
                await self.client.pull(self.model_name)
                logger.info(f"Model '{self.model_name}' pulled successfully.")
            else:
                logger.debug(f"Model '{self.model_name}' already available.")

        except Exception as e:
            logger.error(
                f"Failed to check/pull model '{self.model_name}': {e}")

    def resolve_model(self, override):
        return self.model_name if override == "default" else override

    async def generate(self, messages, **kwargs):
        model_name = self.resolve_model(kwargs.pop("model", "default"))
        stream = kwargs.get("stream", False)
        print(messages)
        async with self.semaphore:
            if stream:
                generator = await self.client.chat(model=model_name, messages=messages, **kwargs)
                return generator
            else:
                response = await self.client.chat(model=model_name, messages=messages, **kwargs)
                return response['message']['content']

    async def expand(self, query: str,
                     expand_instruct: str = ollama_config['expand_instruct'],
                     **kwargs):
        message = {'role': 'user', 'content': f"{expand_instruct} {query}"}
        async with self.semaphore:
            response = await self.generate(messages=[message], **kwargs)
        return response

    async def parse(self, query: str,
                    parse_instruct: str = ollama_config['parse_instruct'],
                    **kwargs):
        message = {'role': 'user', 'content': f"{parse_instruct} {query}"}
        async with self.semaphore:
            response = await self.generate(messages=[message], options={'temperature': 0.0})
        result = extract_parsed_fields(response)
        result['original_query'] = query
        return result

    async def expand_parse(self,
                           query: str,
                           expand_instruct: str = ollama_config['expand_instruct'],
                           parse_instruct: str = ollama_config['parse_instruct'],
                           **kwargs):
        expanded = await self.expand(query, expand_instruct=expand_instruct, **kwargs)
        result = await self.parse(expanded, parse_instruct=parse_instruct, **kwargs)
        return result

    async def expand_parse_batch(self,
                                 queries: List[str],
                                 query_ids: List[int] = None,
                                 expand_instruct: str = ollama_config['expand_instruct'],
                                 parse_instruct: str = ollama_config['parse_instruct'],
                                 **kwargs):
        if not query_ids:
            query_ids = [k for k in range(len(queries))]

        tasks = [self.expand_parse(query=Qstr,
                                   expand_instruct=expand_instruct,
                                   parse_instruct=parse_instruct,
                                   **kwargs)
                 for Qstr in queries]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = {}
        for qid, res in zip(query_ids, results):
            if isinstance(res, Exception):
                logger.error(f"Query {qid} failed: {res}")
            else:
                final_results[qid] = res
        return final_results

    async def generate_batch(self,
                             queries: List[str],
                             query_ids: List[int] = None,
                             context_msg: List[List[dict]] = None,
                             **kwargs):
        if not query_ids:
            query_ids = [k for k in range(len(queries))]
        if not context_msg:
            context_msg = [[] for _ in range(len(queries))]

        query_msgs = [{'role': 'user', 'content': f"{Qstr}"}
                      for Qstr in queries]
        tasks = [self.generate(messages=Cmsg + [Qmsg], **kwargs)
                 for Qmsg, Cmsg in zip(query_msgs, context_msg)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = {}
        for qid, res in zip(query_ids, results):
            if isinstance(res, Exception):
                logger.error(f"Query {qid} failed: {res}")
            else:
                final_results[qid] = res
        return final_results
