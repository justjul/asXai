"""
asXai LLM Inference Module
--------------------------

Manages all interactions with LLM backends (Ollama, Groq) for:
- Model readiness checks and pulls
- Synchronous and streaming chat completions
- Utility methods for planning, writing sections, titles, summaries, query expansion, etc.
- Token buffering to files or asyncio queues for SSE or streaming UIs

Key Components:
- InferenceManager: high-level API for generation and specialized tasks (plan, summarize, expand, etc.)
"""

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

from tqdm import tqdm
from asxai.utils import load_params
from asxai.utils import running_inside_docker

from asxai.logger import get_logger

# Import custom MCP prompt/parse classes
from .mcp_library import (
    QueryParseMCP, ExpandQueryMCP, KeywordsMCP,
    NotebookTitleMCP, ChatSummarizerMCP,
    RelevantPaperMCP, GenerationPlannerMCP,
    parse_model_response
)

# Initialize logger
logger = get_logger(__name__, level=config.LOG_LEVEL)

# Load LLM and chat configuration
params = load_params()
llm_config = params["llm"]
chat_config = params["chat"]


class InferenceManager:
    """
    Orchestrates LLM interactions via Ollama (local) or Groq (cloud).
    Provides methods for:
      - Streaming tokens to file or queue (for SSE)
      - Single-shot generate() calls
      - Specialized routines: generatePlan, writer, generateTitle, chatSummarize, processQuery, expand, keywords, parse, filterArticles
    """

    def __init__(
        self,
        model_name: str = llm_config["model_name"],
        client_timeout: float = llm_config["client_timeout"],
        watch_tmp_timeout: float = llm_config["watch_tmp_timeout"],
        max_threads: int = llm_config["max_threads"]
    ):
        # Core settings
        self.model_name = model_name
        self.client_timeout = client_timeout
        self.watch_tmp_timeout = watch_tmp_timeout
        self.max_threads = max_threads

        self.model_list = llm_config["model_list"]

        # Determine host for Ollama (container vs local)
        self.ollama_port = llm_config["ollama_port"]
        self.ollama_host = "ollama" if running_inside_docker(
        ) else llm_config["host"]

        # Ensure Ollama service is up
        self.is_ollama_running()

        # Initialize asynchronous clients
        self.client_ollama = AsyncClient(
            host=f"http://{self.ollama_host}:{self.ollama_port}",
            timeout=client_timeout
        )
        self.client_groq = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=client_timeout,
            max_retries=5
        )

        # Limit concurrent calls
        self.semaphore = asyncio.Semaphore(self.max_threads)

    def is_ollama_running(self) -> None:
        """
        Polls the Ollama HTTP endpoint until it responds 200 or timeout.
        Logs status and errors.
        """
        ready = False
        url = f"http://{self.ollama_host}:{self.ollama_port}"
        endtime = time.time() + self.client_timeout
        n_retry = 0

        while time.time() < endtime:
            try:
                response = requests.get(url, timeout=2.0)
                if response.status_code == 200:
                    if n_retry > 0:
                        logger.info("Ollama service is up and running.")
                    ready = True
                    return
            except requests.exceptions.RequestException:
                pass
            logger.debug("Ollama not ready yet. Retrying in 3 seconds...")
            n_retry += 1
            time.sleep(3)

        logger.error(f"Ollama did not respond within {self.client_timeout}s.")

    async def is_model_pulled(self) -> None:
        """
        Ensures the specified Ollama model is downloaded locally.
        Pulls if not present.
        """
        try:
            if 'ollama' in self.model_name:
                model_name = self.model_name.split('/', 1)[-1]
                ollama_model_list = await self.client_ollama.list()
                models = ollama_model_list.models
                available_ollama_models = [model.model for model in models]

                if model_name not in available_ollama_models:
                    logger.info(f"Pulling model '{model_name}'...")
                    await self.client_ollama.pull(model_name)
                    logger.info(f"Model '{model_name}' pulled successfully.")
                else:
                    logger.debug(f"Model '{model_name}' already available.")

        except Exception as e:
            logger.error(
                f"Failed to check/pull model '{self.model_name}': {e}")

    def resolve_model(self, override: str) -> Tuple[str, str]:
        """
        Resolves provider and model ID from override or default.
        Validates against configured model list.
        """
        model_name = self.model_name if override == "default" else override
        if model_name not in llm_config.get("model_list", []):
            model_name = self.model_name
        [provider, model_id] = model_name.split('/', 1)
        return (provider, model_id)

    async def stream_to_file(
        self,
        streamer,
        stream_path: Union[str, Path]
    ) -> dict:
        """
        Consume an async token streamer, write tokens to file in chunks,
        and return full 'content' and internal 'think' segments.
        """
        full, think, buffer = "", "", ""
        inside_think = False

        async for chunk in streamer:
            if hasattr(chunk, "message"):
                token = chunk["message"]["content"]
            elif hasattr(chunk, "choices"):
                token = chunk.choices[0].delta.content or ""

            # Handle <think> markers
            if "<think>" in token:
                inside_think = True
                continue
            if "</think>" in token:
                inside_think = False
                continue

            if not inside_think:
                buffer += token
                full += token
            else:
                think += token

            # Flush buffer periodically
            if len(buffer) > 100:
                with open(stream_path, "a") as f:
                    f.write(buffer)
                    f.flush()
                buffer = ""

        # Write any remaining buffer
        if len(buffer) > 0:
            with open(stream_path, "a") as f:
                f.write(buffer)
                f.flush()

        return {'content': full, 'think': think}

    async def stream_to_queue(
        self,
        streamer,
        queue: asyncio.Queue
    ) -> None:
        """
        Similar to stream_to_file, but pushes token dicts into an asyncio.Queue
        for real-time consumption (e.g., SSE).
        """
        buffer, think_buffer = "", ""
        inside_think = False
        async for chunk in streamer:
            if hasattr(chunk, "message"):
                token = chunk["message"]["content"]
            elif hasattr(chunk, "choices"):
                token = chunk.choices[0].delta.content or ""

            if "<think>" in token:
                inside_think = True
                continue
            if "</think>" in token:
                inside_think = False
                continue

            if not inside_think:
                buffer += token
            else:
                think_buffer += token

            if len(buffer) > 100 or len(think_buffer) > 100:
                await queue.put({'content': buffer, 'think': think_buffer})
                buffer = ""
                think_buffer = ""

        # Write any remaining buffer
        if len(buffer) > 0 or len(think_buffer) > 0:
            await queue.put({'content': buffer, 'think': think_buffer})

        # Sentinel
        await queue.put(None)

    async def response_to_queue(
        self,
        coro,
        queue: asyncio.Queue
    ) -> None:
        """
        Awaits a coroutine that returns a single response, then puts it in the queue.
        Captures exceptions as queue items.
        """
        try:
            response = await coro
            await queue.put(response)
        except Exception as e:
            await queue.put(e)

    async def generate(
        self, messages: List[dict],
        **kwargs
    ) -> Union[str, AsyncGenerator]:
        """
        High-level chat call. Supports Ollama or Groq providers, streaming or not.

        Args:
            messages: List[{"role": "...", "content": "..."}]
            kwargs: includes 'model', 'stream', and any provider-specific options (e.g. temperature).

        Returns:
            If stream=True, returns an async generator. Else, returns a string response.
        """
        provider, model_name = self.resolve_model(
            kwargs.pop("model", "default"))
        stream = kwargs.pop("stream", False)

        # Ollama path
        if provider.lower() == "ollama":
            options = kwargs
            async with self.semaphore:
                if stream:
                    streamer = await self.client_ollama.chat(
                        model=model_name, messages=messages, stream=True, options=options
                    )
                    return streamer
                else:
                    response = await self.client_ollama.chat(
                        model=model_name, messages=messages, stream=False, options=options
                    )
                    return response['message']['content']
        # Groq path
        if provider.lower() == "groq":
            async with self.semaphore:
                if stream:
                    streamer = await self.client_groq.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=True,
                        **kwargs
                    )
                    return streamer
                else:
                    response = await self.client_groq.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        **kwargs
                    )
                    return response.choices[0].message.content

    # ----- Specialized MCP-based methods -----

    async def generatePlan(
        self, query: str,
        documents: List[str] = None,
        genplan_instruct: str = chat_config['instruct_genplan'],
        **kwargs
    ) -> List[dict]:
        """
        Builds a multi-step plan from a query, using the GenerationPlannerMCP.
        """
        instruct = GenerationPlannerMCP.generate_prompt(genplan_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': doc}
                    for doc in (documents or [])]
        messages += [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)

        result = GenerationPlannerMCP.parse(response)

        return result

    async def writer(
        self, title: str = '',
        content: str = None,
        documents: List[str] = None,
        context: List[dict] = None,
        instruct: str = chat_config['instruct_gensection'],
        **kwargs
    ):
        """
        Generates a structured section of content based on title/content templates + documents/context.
        """
        prompt = instruct.replace("<TITLE>", title)
        prompt = instruct.replace("<CONTENT>", content)

        messages = []
        if context:
            messages.extend(
                [{'role': msg['role'], 'content': msg['content']}
                    for msg in context]
            )
        messages.extend([{'role': 'user', 'content': doc}
                        for doc in (documents or [])])
        messages.extend([{'role': 'user', 'content': prompt}])

        if kwargs.pop("stream", False):
            return await self.generate(messages=messages, stream=True, **kwargs)

        response = await self.generate(messages=messages, stream=False, **kwargs)
        res = parse_model_response(response)
        return res

    async def generateTitle(
        self, query: str,
        title_instruct: str = chat_config['instruct_title'],
        **kwargs
    ) -> dict:
        """
        Generates a notebook title suggestion from a query.
        """
        instruct = NotebookTitleMCP.generate_prompt(title_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)
        result = NotebookTitleMCP.parse(response)

        return result

    async def chatSummarize(
        self,
        chat_history: list[dict],
        chatSummary_instruct:  str = chat_config['instruct_chatSummary'],
        summary_len: int = chat_config['summary_length'],
        **kwargs
    ) -> str:
        """
        Produces a concise summary of recent chat history.
        """
        instruct = ChatSummarizerMCP.generate_prompt(chatSummary_instruct)
        instruct = instruct.replace("<SUMMARY_LENGTH>", str(summary_len))

        # Flatten history into two sections
        prev, recent = "- PREVIOUS SUMMARIES:\n", "- RECENT HISTORY:\n"
        for turn in chat_history:
            if turn["model"] != "search-worker":
                content = turn["content"]
                if turn["model"] == "summarizer":
                    prev += f"  + {content}\n"
                else:
                    recent += f"  + {turn['role'].capitalize()}: {content}\n"

        prompt = instruct.replace("<HISTORY>", prev + recent)

        messages = [{'role': 'user', 'content': prompt}]

        temperature = kwargs.pop('temperature', 0.0)
        response = await self.generate(messages=messages, temperature=temperature, **kwargs)
        res = parse_model_response(response)

        return res['content']

    async def processQuery(
        self,
        query: str,
        chat_history: List[dict],
        expand_instruct: str = chat_config['instruct_expand'],
        **kwargs
    ) -> dict:
        """
        Full query processing pipeline:
          1) Expand the user query into refined questions.
          2) Optionally extract keywords and parse query structure.
          3) Combine everything into a single result dict.

        Args:
            query: Original user query.
            chat_history: List of prior messages for context.
            expand_instruct: Instruction template for query expansion.
            **kwargs: Passed to generate() for model options.

        Returns:
            Dict containing expanded queries, keywords, parsed fields, and
            any search_paperIds identified.
        """
        # Build expansion prompt and messages
        instruct = ExpandQueryMCP.generate_prompt(expand_instruct)
        prompt = instruct.replace("<QUERY>", query)

        context = [{'role': msg['role'], 'content': msg['content']}
                   for msg in chat_history]
        messages = context + [{'role': 'user', 'content': prompt}]

        # Call LLM to expand query
        response = await self.generate(messages=messages, **kwargs)
        result = ExpandQueryMCP.parse(response)
        logger.info(f"Expand results: {result}")

        queries = result.get('queries', [])
        scientific = result.get('scientific', [])
        ethical = result.get('ethical', [])
        details = result.get('details', [])
        if not details:
            result['search_paperIds'] = []

        # We concatenate questions to deal with a single search to parse.
        query_to_parse = ' '.join(queries)

        # derive keywords and parse structure if question is valid
        if scientific and ethical:
            payload = await self.keywords(query=query_to_parse, **kwargs)
            parsed = await self.parse(query_to_parse, **kwargs)
        else:
            payload, parsed = {}, {}

        # If expansion yields paper IDs from context, skip parsing
        if result['search_paperIds']:
            parsed = {}

        # Merge all results
        results = {**result, **payload, **parsed}
        logger.info(f"Expand + Keywords + Parsed results: {results}")

        return results

    async def expand(
        self, query: str,
        chat_history: List[dict],
        expand_instruct: str = chat_config['instruct_expand'],
        **kwargs
    ) -> List[dict]:
        """
        Query expansion, identification of explicitely referred papers, 
        ethical check, etc using ExpandQueryMCP

        Args:
            query: Original user query.
            chat_history: Prior conversation for context.
            expand_instruct: Instruction template for expansion.
            **kwargs: Passed to generate().

        Returns:
            List of dicts with expanded 'queries', 'scientific', 'ethical', searched_paperIds, etc.
        """
        instruct = ExpandQueryMCP.generate_prompt(expand_instruct)
        prompt = instruct.replace("<QUERY>", query)

        context = [{'role': msg['role'], 'content': msg['content']}
                   for msg in chat_history]
        messages = context + [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)
        results = ExpandQueryMCP.parse(response)

        return results

    async def keywords(
        self, query: str,
        keyword_instruct: str = chat_config['instruct_keyword'],
        **kwargs
    ) -> dict:
        """
        Extracts relevant keywords from the query using the KeywordsMCP.

        Args:
            query: The user's question.
            keyword_instruct: Instruction template for keyword extraction.
            **kwargs: Passed to generate().

        Returns:
            Dict of extracted keywords, e.g. {'keywords': [...]}.
        """
        instruct = KeywordsMCP.generate_prompt(keyword_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': prompt}]

        response = await self.generate(messages=messages, **kwargs)
        result = KeywordsMCP.parse(response)

        return result

    async def parse(
        self, query: str,
        parse_instruct: str = chat_config['instruct_parse'],
        **kwargs
    ) -> dict:
        """
        Parses the query into structured fields using QueryParseMCP
        (e.g., authors, date ranges, venue, article type).

        Args:
            query: The user's question.
            parse_instruct: Instruction template for parsing.
            **kwargs: Passed to generate().

        Returns:
            Dict of parsed parameters for downstream search.
        """
        instruct = QueryParseMCP.generate_prompt(parse_instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = [{'role': 'user', 'content': prompt}]

        temperature = kwargs.pop('temperature', 0.0)
        response = await self.generate(messages=messages, temperature=temperature, **kwargs)
        result = QueryParseMCP.parse(response)

        return result

    async def filterArticles(
        self,
        query: str = None,
        documents: List[str] = None,
        instruct: str = chat_config['instruct_paperfilter'],
        **kwargs
    ) -> List[str]:
        """
        Filters a set of candidate papers based on relevance to the query, 
        using RelevantPaperMCP.

        Args:
            query: The user's original query (optional).
            documents: List of document text snippets to evaluate.
            instruct: Instruction template for filtering.
            **kwargs: Passed to generate() (e.g., temperature).

        Returns:
            List of paper IDs or titles deemed relevant.
        """
        # Build prompt combining docs + filtering instruction
        instruct = RelevantPaperMCP.generate_prompt(instruct)
        prompt = instruct.replace("<QUERY>", query)

        messages = []
        messages.extend([{'role': 'user', 'content': doc}
                        for doc in (documents or [])])
        messages.extend([{'role': 'user', 'content': prompt}])

        temperature = kwargs.pop('temperature', 0.0)
        response = await self.generate(messages=messages, temperature=temperature, **kwargs)
        results = RelevantPaperMCP.parse(response)
        logger.info(f"Article filter response after parsing: {results}")
        return results
