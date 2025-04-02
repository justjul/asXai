import os
import config
import docker
import docker.errors
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff
from qdrant_client.models import TokenizerType, TextIndexParams
from qdrant_client.models import Filter, FieldCondition
from qdrant_client.models import MatchValue, MatchAny
from qdrant_client.models import WithLookup

from typing import List, Tuple, Union, TypedDict, AsyncGenerator
from pathlib import Path
import json
import uuid
import time
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
        self.tmp_path = config.VECTORDB_PATH / "tmp"
        self.container_name = config.PROJECT_NAME + "-Qdrant"
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

        self.start_qdrant_container()
        self.client = AsyncQdrantClient(host=self.host, port=self.port)

    def start_qdrant_container(self):
        client = docker.from_env()
        try:
            container = client.containers.get(self.container_name)
            if container.status != "running":
                logger.info("Starting existing Qdrant container...")
                container.start()
            else:
                logger.info("Qdrant container already up and running")
        except docker.errors.NotFound:
            logger.info(
                "No Qdrant container running. Starting a new one now...")
            container = client.containers.run(image="qdrant/qdrant",
                                              name=self.container_name,
                                              ports={
                                                  f"{self.port}/tcp": self.port},
                                              volumes={
                                                  str(self.volume_path): {
                                                      'bind': '/qdrant/storage',
                                                      'mode': 'rw'
                                                  }
                                              },
                                              detach=True)

        ready = False
        endtime = time.time() + self.docker_timeout
        while time.time() < endtime:
            time.sleep(3)
            try:
                container = client.containers.get(self.container_name)
                ready = True
                break
            except docker.errors.NotFound:
                pass
        if not ready:
            logger.error("Qdrant container didn't start before time out")

    def stop_qdrant_container(self):
        client = docker.from_env()
        try:
            logger.info("Will try to stop Qdrant container...")
            container = client.containers.get(self.container_name)
            container.stop()
            container.remove()
            logger.info("Qdrant container stopped and removed...")
        except docker.errors.NotFound:
            logger.info("Qdrant container not found...")

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
        chunk_points = [PointStruct(id=str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=f"{paperId}_{i}")),
                                    vector=emb, payload={field: payloads[i][field] for field in self.chunk_fields})
                        for i, emb in enumerate(embeddings)]

        await self.client.upsert(collection_name=self.collection_name_chunks, points=chunk_points)
        try:
            id_points = [PointStruct(id=str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=f"{paperId}")),
                                     vector=ref_embedding, payload={field: payloads[0][field] for field in self.id_fields})]
        except Exception as e:
            print(paperId)
            raise e
        await self.client.upsert(collection_name=self.collection_name_ids, points=id_points)

        if self.clean_after_push:
            os.remove(file_path)

    async def check_collection(self):
        if not await self.client.collection_exists(self.collection_name_ids):
            await self.client.create_collection(
                collection_name=self.collection_name_ids,
                vectors_config=VectorParams(size=self.vector_size,
                                            distance=Distance.COSINE),
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
                hnsw_config=HnswConfigDiff(on_disk=True,
                                           m=16,
                                           ef_construct=100))

        if not await self.client.collection_exists(self.collection_name_chunks):
            await self.client.create_collection(
                collection_name=self.collection_name_chunks,
                vectors_config=VectorParams(size=self.vector_size,
                                            distance=Distance.COSINE),
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
                hnsw_config=HnswConfigDiff(on_disk=True,
                                           m=16,
                                           ef_construct=100))

    async def sync_qdrant(self, done_event=None):
        if done_event is None:
            class DummyEvent:
                def is_set(self): return True
            done_event = DummyEvent()

        self.semaphore = asyncio.Semaphore(self.max_threads)

        await self.check_collection()

        new_ids = self.get_embedding_ids()
        pushed_ids = []
        while new_ids or not done_event.is_set():
            paper_ids = self.get_embedding_ids()
            new_ids = [id for id in paper_ids if id not in pushed_ids]
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

        self.stop_qdrant_container()

        if os.path.isdir(self.tmp_path) and self.clean_after_push:
            os.removedirs(self.tmp_path)

    async def query(self,
                    query_vector: List[float],
                    topK: int = 5,
                    topK_per_paper: int = 5,
                    filter_by_paper_ids: Union[List[str], None] = None,
                    filter_payloads: Union[dict, None] = None):

        if not hasattr(self, 'client'):
            self.client = AsyncQdrantClient(host=self.host, port=self.port)

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
            collection_name=self.collection_name_chunks,
            query=query_vector,
            query_filter=query_filter,
            group_by="paperId",
            limit=topK,  # Max amount of groups
            group_size=topK_per_paper,  # Max amount of points per group
            with_lookup=WithLookup(
                collection=self.collection_name_ids,
                with_payload=["title", "abstract", "text"],
                with_vectors=False,),
            with_payload=True)

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
