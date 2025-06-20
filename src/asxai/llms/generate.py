import os
import config
import asyncio

from ollama import AsyncClient
from groq import AsyncGroq

import requests

from typing import List, Tuple, Union, TypedDict, AsyncGenerator
from pathlib import Path
import uuid
import time
from dateutil.parser import parse
from datetime import datetime

from .mcp_library import QueryParseMCP, ExpandQueryMCP, KeywordsMCP, NotebookTitleMCP, ChatSummarizerMCP
from .mcp_library import QuickReplyMCP, GenerationPlannerMCP, SectionGenerationMCP

from tqdm import tqdm
from asxai.utils import load_params
from asxai.utils import running_inside_docker

from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
embedding_config = params["embedding"]
qdrant_config = params["qdrant"]
ollama_config = params["ollama"]
chat_config = params["chat"]


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
        self.client_ollama = AsyncClient(
            host=f"http://{self.host}:{self.port}", timeout=120.0
        )
        self.client_groq = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY"), timeout=120.0, max_retries=5
        )
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
            model_list = await self.client_ollama.list()
            models = model_list.models
            available_models = [model.model for model in models]

            if self.model_name not in available_models:
                logger.info(f"Pulling model '{self.model_name}'...")
                await self.client_ollama.pull(self.model_name)
                logger.info(f"Model '{self.model_name}' pulled successfully.")
            else:
                logger.debug(f"Model '{self.model_name}' already available.")

        except Exception as e:
            logger.error(
                f"Failed to check/pull model '{self.model_name}': {e}")

    def resolve_model(self, override):
        return self.model_name if override == "default" else override

    async def stream(self, streamer, stream_path, provider=chat_config['inference_service']):
        full_response = ""
        buffer = ""
        async for chunk in streamer:
            if provider.lower() == "ollama":
                token = chunk["message"]["content"]
            elif provider.lower() == "groq":
                token = chunk.choices[0].delta.content or ""
            buffer += token

            if len(buffer) > 100:
                with open(stream_path, "a") as stream_file:
                    stream_file.write(buffer)
                    stream_file.flush()
                buffer = ""
            full_response += token

        # Write any remaining buffer
        if len(buffer) > 0:
            with open(stream_path, "a") as stream_file:
                stream_file.write(buffer)
                stream_file.flush()

        return full_response

    async def generate(self, messages, provider=chat_config['inference_service'], **kwargs):
        model_name = self.resolve_model(kwargs.pop("model", "default"))
        stream = kwargs.pop("stream", False)
        if provider.lower() == "ollama":
            options = kwargs
            async with self.semaphore:
                if stream:
                    streamer = await self.client_ollama.chat(model=model_name, messages=messages, stream=True, options=options)
                    return streamer
                else:
                    response = await self.client_ollama.chat(model=model_name, messages=messages, stream=False, options=options)
                    return response['message']['content']
        elif provider.lower() == "groq":
            async with self.semaphore:
                if stream:
                    streamer = await self.client_groq.chat.completions.create(
                        # "meta-llama/llama-4-scout-17b-16e-instruct",  # "llama3-8b-8192",#model_name
                        model="llama3-70b-8192",
                        messages=messages,
                        stream=True,
                        **kwargs
                    )
                    return streamer
                else:
                    response = await self.client_groq.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",  # "llama3-8b-8192",#model_name
                        messages=messages,
                        stream=False,
                        **kwargs
                    )
                    return response.choices[0].message.content

    async def generateQuickReply(self, query: str,
                                 documents: str,
                                 quick_instruct: str = chat_config['instruct_quickreply'],
                                 **kwargs):
        instruct = QuickReplyMCP.generate_prompt(quick_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': doc} for doc in documents]
        messages += [{'role': 'user', 'content': prompt}]

        kwargs.pop("stream", None)
        streamer = await self.generate(messages=messages, stream=True, **kwargs)

        async for chunk in streamer:
            yield chunk

    async def generatePlan(self, query: str,
                           documents: str,
                           genplan_instruct: str = chat_config['instruct_genplan'],
                           **kwargs) -> List[dict]:

        # articles_serialized = "- ARTICLES:\n" + '\n'.join(documents)
        # query_serialized = "- QUERY:\n" + query

        # instruct = GenerationPlannerMCP.generate_prompt(genplan_instruct)
        # prompt = instruct.replace(
        #     "<QUERY>", articles_serialized + query_serialized)

        instruct = GenerationPlannerMCP.generate_prompt(genplan_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': doc} for doc in documents]
        messages += [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)

        result = GenerationPlannerMCP.parse(response)

        return result

    async def generateSection(self, title: str, content: str,
                              documents: str,
                              gensection_instruct: str = chat_config['instruct_gensection'],
                              **kwargs):

        instruct = SectionGenerationMCP.generate_prompt(gensection_instruct)
        prompt = instruct.replace("<TITLE>", title)
        prompt = instruct.replace("<CONTENT>", content)

        messages = [{'role': 'user', 'content': doc} for doc in documents]
        messages += [{'role': 'user', 'content': prompt}]

        kwargs.pop("stream", None)
        streamer = await self.generate(messages=messages, stream=True, **kwargs)

        async for chunk in streamer:
            yield chunk

    async def generateTitle(self, query: str,
                            title_instruct: str = chat_config['instruct_title'],
                            **kwargs) -> dict:
        instruct = NotebookTitleMCP.generate_prompt(title_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)
        result = NotebookTitleMCP.parse(response)

        return result

    async def chatSummarize(self, chat_history: list[dict],
                            chatSummary_instruct:  str = chat_config['instruct_chatSummary'],
                            summary_len: int = chat_config['summary_length']) -> str:
        instruct = ChatSummarizerMCP.generate_prompt(chatSummary_instruct)
        instruct = instruct.replace("<SUMMARY_LENGTH>", str(summary_len))

        summaries_serialized = "- PREVIOUS SUMMARIES:\n"
        chat_serialized = "- RECENT HISTORY:\n"
        for turn in chat_history:
            if turn['model'] != 'search-worker':
                if turn['model'] != 'summarizer':
                    role = turn['role'].capitalize()
                    content = turn['content']
                    chat_serialized += f"    + {role}: {content}\n"
                else:
                    content = turn['content']
                    summaries_serialized += f"    + {content}\n"

        prompt = instruct.replace(
            "<HISTORY>", summaries_serialized + chat_serialized)

        messages = [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, temperature=0.0)
        # result = parse_mcp_response(response)

        return response

    async def processQuery(self, query: str, chat_history: List[dict],
                           expand_instruct: str = chat_config['instruct_expand'],
                           **kwargs):
        instruct = ExpandQueryMCP.generate_prompt(expand_instruct)
        prompt = instruct.replace("<QUERY>", query)

        context = [{'role': msg['role'], 'content': msg['content']}
                   for msg in chat_history]
        messages = context + [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)
        result = ExpandQueryMCP.parse(response)

        logger.info(f"Expand results: {result}")

        queries = result.get('queries', [])
        search_needed = result.get('search_needed', [])

        # This is a temporary fix before we adjust search strategy with multiple questions:
        # We concatenate questions to deal with a single search per user's query.
        query_to_parse = ' '.join(queries)

        results = []
        if search_needed:
            payload = await self.keywords(query=query_to_parse, **kwargs)
            parsed = await self.parse(query_to_parse, **kwargs)
        else:
            payload, parsed = {}, {}

        results = {"queries": queries, **payload, **parsed}

        logger.info(f"Expand + Keywords + Parsed results: {results}")

        return results

    async def expand(self, query: str, chat_history: List[dict],
                     expand_instruct: str = chat_config['instruct_expand'],
                     **kwargs):
        instruct = ExpandQueryMCP.generate_prompt(expand_instruct)
        prompt = instruct.replace("<QUERY>", query)

        context = [{'role': msg['role'], 'content': msg['content']}
                   for msg in chat_history]
        messages = context + [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)
        results = ExpandQueryMCP.parse(response)

        return results

    async def keywords(self, query: str,
                       keyword_instruct: str = chat_config['instruct_keyword'],
                       **kwargs):
        instruct = KeywordsMCP.generate_prompt(keyword_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)
        result = KeywordsMCP.parse(response)

        return result

    async def parse(self, query: str,
                    parse_instruct: str = chat_config['instruct_parse'],
                    **kwargs):

        instruct = QueryParseMCP.generate_prompt(parse_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, temperature=0.0, **kwargs)
        result = QueryParseMCP.parse(response)

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
