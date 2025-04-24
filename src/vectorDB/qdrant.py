import os
import config
import docker
import docker.errors
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, Datatype, VectorParams, PointStruct
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff
from qdrant_client.models import TokenizerType, TextIndexParams
from qdrant_client.models import Filter, FieldCondition
from qdrant_client.models import MatchValue, MatchAny
from qdrant_client.models import WithLookup
import requests

from typing import List, Tuple, Union, TypedDict, AsyncGenerator
from pathlib import Path
import json
import uuid
import time
import math
from tqdm import tqdm
from transformers import AutoConfig
from src.utils import load_params

import logging
from src.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

params = load_params()
embedding_config = params["embedding"]
qdrant_config = params["qdrant"]
qdrant_config["collection_name"] = embedding_config["model_name"]


class QdrantManager:
    def __init__(self,
                 model_name: str = qdrant_config["model_name"],
                 docker_timeout: float = qdrant_config["docker_timeout"],
                 watch_tmp_timeout: float = qdrant_config["watch_tmp_timeout"],
                 max_threads: int = qdrant_config["max_threads"],
                 clean_after_push: bool = True):

        self.volume_path = config.VECTORDB_PATH / "Qdrant"
        if not self.volume_path.exists():
            self.volume_path.mkdir(parents=True, exist_ok=True)
        self.tmp_path = config.TMP_PATH / "embeddings"
        self.container_name = "qdrant"
        self.collection_name = model_name.split('/')[-1]
        self.model_name = model_name

        self.collection_name_chunks = self.collection_name + '_chunks'
        self.collection_name_ids = self.collection_name + '_ids'
        self.chunk_fields = {'paperId', 'fieldsOfStudy', 'venue', 'authorName',
                             'citationCount', 'influentialCitationCount',
                             'publicationDate', 'publicationYear', 'text'}
        self.id_fields = {'paperId', 'fieldsOfStudy', 'venue', 'authorName',
                          'citationCount', 'influentialCitationCount',
                          'publicationDate', 'publicationYear', 'openAccessPdf',
                          'title', 'abstract', 'main_text'}

        hidden_size = AutoConfig.from_pretrained(self.model_name).hidden_size
        self.vector_size = hidden_size

        self.port = qdrant_config["port"]
        self.host = qdrant_config["host"]

        self.docker_timeout = docker_timeout
        self.watch_tmp_timeout = watch_tmp_timeout
        self.max_threads = max_threads
        self.clean_after_push = clean_after_push

        self.is_qdrant_running()
        self.client = AsyncQdrantClient(host=self.host,
                                        port=self.port,
                                        timeout=120.0)
        self.semaphore = asyncio.Semaphore(self.max_threads)

    def is_qdrant_running(self):
        logger.info(f"Waiting for Qdrant container '{self.container_name}'...")
        ready = False
        endtime = time.time() + self.docker_timeout
        url = f"http://{self.host}:{self.port}/collections"

        while time.time() < endtime:
            try:
                response = requests.get(url, timeout=2.0)
                if response.status_code == 200:
                    logger.info("Qdrant service is up and running.")
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            logger.debug("Qdrant not ready yet. Retrying in 3 seconds...")
            time.sleep(3)

        if not ready:
            logger.error(
                f"Qdrant container '{self.container_name}' did not become ready within timeout.")

    # # TO BE REMOVED
    #     client = docker.from_env()
    #     try:
    #         container = client.containers.get(self.container_name)
    #         if container.status != "running":
    #             logger.info("Starting existing Qdrant container...")
    #             container.start()
    #         else:
    #             logger.info("Qdrant container already up and running")
    #     except docker.errors.NotFound:
    #         logger.info(
    #             "No Qdrant container running. Starting a new one now...")
    #         n_attempt = 0
    #         while n_attempt < 10:
    #             n_attempt += 1
    #             try:
    #                 container = client.containers.run(image="qdrant/qdrant",
    #                                                   name=self.container_name,
    #                                                   ports={
    #                                                       f"{self.port}/tcp": self.port},
    #                                                   volumes={
    #                                                       str(self.volume_path): {
    #                                                           'bind': '/qdrant/storage',
    #                                                           'mode': 'rw'
    #                                                       }
    #                                                   },
    #                                                   detach=True)
    #             except Exception:
    #                 time.sleep(5)

    #     ready = False
    #     endtime = time.time() + self.docker_timeout
    #     while time.time() < endtime:
    #         time.sleep(3)
    #         try:
    #             container = client.containers.get(self.container_name)
    #             ready = True
    #             break
    #         except docker.errors.NotFound:
    #             pass
    #     if not ready:
    #         logger.error("Qdrant container didn't start before time out")

    # def stop_qdrant_container(self):
    #     client = docker.from_env()
    #     try:
    #         logger.info("Will try to stop Qdrant container...")
    #         container = client.containers.get(self.container_name)
    #         container.stop()
    #         container.remove()
    #         logger.info("Qdrant container stopped and removed...")
    #     except docker.errors.NotFound:
    #         logger.info("Qdrant container not found...")

    # # TO BE REMOVED

    async def get_existing_paperIds(self):
        paperIds = set()
        scroll_cursor = None

        while True:
            scroll_result, scroll_cursor = await self.client.scroll(
                collection_name=self.collection_name_ids,
                scroll_filter=None,
                with_payload='paperId',
                with_vectors=False,
                limit=1000,  # smaller batch size for pagination
                offset=scroll_cursor
            )
            for point in scroll_result:
                paperIds.add(point['payload']['paperId'])
            if scroll_cursor is None:
                break

        return paperIds

    def get_embedding_ids(self):
        files = list(Path(self.tmp_path).glob("*.json"))
        return [f.stem for f in files]

    async def delete_papers(self, paperIds):
        delete_tasks = [
            self.client.delete(
                collection_name=self.collection_name_chunks,
                points_selector=Filter(must=[FieldCondition(key="paperId",
                                                                match=MatchAny(any=paperIds))])
            )
            # for id in paperIds
        ]
        delete_tasks.extend([
            self.client.delete(
                collection_name=self.collection_name_ids,
                points_selector=Filter(must=[FieldCondition(key="paperId",
                                                                match=MatchAny(any=paperIds))])
            )
            # for id in paperIds
        ])

        await asyncio.gather(*delete_tasks)

    async def upload_paper(self, paperId):
        file_path = self.tmp_path / f"{paperId}.json"
        with open(file_path, "r") as f:
            data = json.load(f)
        embeddings = data['embeddings']
        payloads = data['payloads']
        ref_embedding = data['ref_embedding']
        paperId_Qrant = str(uuid.uuid5(
            namespace=uuid.NAMESPACE_DNS, name=f"{paperId}"))
        id_points = [PointStruct(id=paperId_Qrant,
                                 vector=ref_embedding,
                                 payload={'paperIdQ': paperId_Qrant,
                                          **{field: payloads[0][field] for field in self.id_fields}})]

        chunk_points = [PointStruct(id=str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=f"{paperId}_{i}")),
                                    vector=emb,
                                    payload={'paperIdQ': paperId_Qrant,
                                             **{field: payloads[i][field] for field in self.chunk_fields}})
                        for i, emb in enumerate(embeddings)]

        max_retries = 5
        backoff_seconds = 5
        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    await self.client.upsert(collection_name=self.collection_name_ids, points=id_points)
                    await self.client.upsert(collection_name=self.collection_name_chunks, points=chunk_points)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                else:
                    logger.warning(
                        f"Upsert failed (attempt {attempt+1}), retrying in {backoff_seconds}s: {e}")
                    await asyncio.sleep(backoff_seconds)

        if self.clean_after_push:
            os.remove(file_path)

    async def check_collection(self):
        if not await self.client.collection_exists(self.collection_name_ids):
            await self.client.create_collection(
                collection_name=self.collection_name_ids,
                vectors_config=VectorParams(size=self.vector_size,
                                            distance=Distance.COSINE,
                                            datatype=Datatype.FLOAT16),
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
                hnsw_config=HnswConfigDiff(on_disk=True,
                                           m=16,
                                           ef_construct=100))

        await self.client.create_payload_index(collection_name=self.collection_name_ids,
                                               field_name="paperId",
                                               field_schema="keyword")

        if not await self.client.collection_exists(self.collection_name_chunks):
            await self.client.create_collection(
                collection_name=self.collection_name_chunks,
                vectors_config=VectorParams(size=self.vector_size,
                                            distance=Distance.COSINE,
                                            datatype=Datatype.FLOAT16),
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
                hnsw_config=HnswConfigDiff(on_disk=True,
                                           m=16,
                                           ef_construct=100))

        await self.client.create_payload_index(collection_name=self.collection_name_chunks,
                                               field_name="paperId",
                                               field_schema="keyword")

    async def sync_qdrant(self, done_event=None):
        if done_event is None:
            class DummyEvent:
                def is_set(self): return True
            done_event = DummyEvent()

        await self.check_collection()

        new_ids = self.get_embedding_ids()
        pushed_ids = []
        while True:
            paper_ids = self.get_embedding_ids()
            new_ids = [id for id in paper_ids if id not in pushed_ids]

            if not new_ids and done_event.is_set():
                break

            if new_ids:
                await self.delete_papers(new_ids)

                upload_tasks = [self.upload_paper(id) for id in new_ids]
                with tqdm(asyncio.as_completed(upload_tasks),
                          total=len(upload_tasks),
                          desc='Uploading') as pbar:
                    for f in pbar:
                        await f

            pushed_ids = paper_ids
            time.sleep(5)

        if os.path.isdir(self.tmp_path) and self.clean_after_push:
            os.removedirs(self.tmp_path)

    async def update_payload(self,
                             new_payload: dict,):

        new_payload_id = {field: new_payload[field]
                          for field in self.id_fields
                          if field in new_payload}
        new_payload_chunks = {field: new_payload[field]
                              for field in self.chunk_fields
                              if field in new_payload}

        paper_id = new_payload_id['paperId']
        match_filter = Filter(must=[
            FieldCondition(
                key="paperId",
                match=MatchValue(value=paper_id)
            )
        ])

        async with self.semaphore:
            await asyncio.gather(
                self.client.set_payload(
                    collection_name=self.collection_name_ids,
                    payload=new_payload_id,
                    points=match_filter
                ),
                self.client.set_payload(
                    collection_name=self.collection_name_chunks,
                    payload=new_payload_chunks,
                    points=match_filter
                )
            )

    async def batch_update_payloads(self, paperdata):
        batch_size = 1000
        with tqdm(range(math.ceil(len(paperdata) / (batch_size + 1))),
                  desc='Updating payloads') as pbar:
            for i in pbar:
                payloads = paperdata[i * batch_size: (i+1)*batch_size]
                tasks = []
                for new_payload in payloads:
                    tasks.append(self.update_payload(new_payload))

                await asyncio.gather(*tasks)

    async def query(self,
                    query_vector: List[float],
                    topK: int = 5,
                    topK_per_paper: int = 5,
                    filter_by_paper_ids: Union[List[str], None] = None,
                    filter_payloads: Union[dict, None] = None,
                    collection_name: str = None,
                    **kwargs):

        if not hasattr(self, 'client'):
            self.client = AsyncQdrantClient(host=self.host, port=self.port)

        if collection_name is None:
            collection_name = self.collection_name_chunks

        must_conditions = []

        if filter_by_paper_ids:
            must_conditions.append(FieldCondition(
                key="paperId", match=MatchAny(any=filter_by_paper_ids)))

        if filter_payloads:
            for key, values in filter_payloads.items():
                must_conditions.append(FieldCondition(
                    key=key, match=MatchAny(any=values)))

        query_filter = Filter(
            must=must_conditions) if must_conditions else None

        results = await self.client.query_points_groups(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            group_by="paperIdQ",
            limit=topK,  # Max amount of groups
            group_size=topK_per_paper,  # Max amount of points per group
            with_lookup=WithLookup(
                collection=self.collection_name_ids,
                with_payload=True),
            with_payload=True,
            **kwargs)

        return results

    async def query_batch_streamed(self,
                                   query_vectors: List[List[float]],
                                   topK: int = 5,
                                   topK_per_paper: int = 5, **kwargs):
        tasks = [
            self.query(query_vector=vec,
                       topK=topK,
                       topK_per_paper=topK_per_paper,
                       **kwargs)
            for vec in query_vectors
        ]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
        return results

    async def query_stream_generator(self,
                                     query_vectors: List[List[float]],
                                     topK: int = 5,
                                     topK_per_paper: int = 5, **kwargs) -> AsyncGenerator:
        tasks = [
            self.query(query_vector=vec,
                       topK=topK,
                       topK_per_paper=topK_per_paper,
                       **kwargs)
            for vec in query_vectors
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            yield result
