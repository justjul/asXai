import os
import config
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, Datatype, VectorParams, PointStruct
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff
from qdrant_client.models import TokenizerType, TextIndexParams
from qdrant_client.models import Filter, FieldCondition
from qdrant_client import models
from qdrant_client.models import WithLookup
import requests

from typing import List, Optional, Tuple, Union, TypedDict, AsyncGenerator
from pathlib import Path
import json
import uuid
import time
import math
from tqdm import tqdm
from transformers import AutoConfig
from asxai.utils import load_params
from asxai.utils import running_inside_docker

from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)

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
        self.default_vector = {self.collection_name_chunks: "default",
                               self.collection_name_ids: "default"}

        self.port = qdrant_config["port"]
        self.host = "qdrant" if running_inside_docker(
        ) else qdrant_config["host"]

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
        ready = False
        endtime = time.time() + self.docker_timeout
        url = f"http://{self.host}:{self.port}/collections"
        n_retry = 0

        while time.time() < endtime:
            try:
                response = requests.get(url, timeout=2.0)
                if response.status_code == 200:
                    if n_retry > 0:
                        logger.info("Qdrant service is up and running.")
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            logger.debug("Qdrant not ready yet. Retrying in 3 seconds...")
            n_retry += 1
            time.sleep(3)

        if not ready:
            logger.error(
                f"Qdrant container '{self.container_name}' did not become ready within timeout.")

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
                                                                match=models.MatchAny(any=paperIds))])
            )
            # for id in paperIds
        ]
        delete_tasks.extend([
            self.client.delete(
                collection_name=self.collection_name_ids,
                points_selector=Filter(must=[FieldCondition(key="paperId",
                                                                match=models.MatchAny(any=paperIds))])
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
        mean_embedding = data['mean_embedding']
        paperId_Qrant = str(uuid.uuid5(
            namespace=uuid.NAMESPACE_DNS, name=f"{paperId}"))
        id_points = [PointStruct(id=paperId_Qrant,
                                 vector=embeddings,  # mean_embedding
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
            vec_config = VectorParams(size=self.vector_size,
                                      distance=Distance.COSINE,
                                      datatype=Datatype.FLOAT16,
                                      multivector_config=models.MultiVectorConfig(
                                          comparator=models.MultiVectorComparator.MAX_SIM))
            await self.client.create_collection(
                collection_name=self.collection_name_ids,
                vectors_config=vec_config,
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
                hnsw_config=HnswConfigDiff(on_disk=True,
                                           m=16,
                                           ef_construct=100))

        await self.client.create_payload_index(collection_name=self.collection_name_ids,
                                               field_name="paperId",
                                               field_schema="keyword")

        if not await self.client.collection_exists(self.collection_name_chunks):
            vec_config = VectorParams(size=self.vector_size,
                                      distance=Distance.COSINE,
                                      datatype=Datatype.FLOAT16)
            await self.client.create_collection(
                collection_name=self.collection_name_chunks,
                vectors_config=vec_config,
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
                match=models.MatchValue(value=paper_id)
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
                    payload_filter: Union[dict, None] = None,
                    collection_name: str = None,
                    **kwargs):

        if not hasattr(self, 'client'):
            self.client = AsyncQdrantClient(host=self.host, port=self.port)

        if collection_name is None:
            collection_name = self.collection_name_ids

        query_filter = build_qdrant_filter(payload_filter=payload_filter)

        results = await self.client.query_points(
            collection_name=self.collection_name_ids,
            query=query_vector,
            query_filter=query_filter,
            limit=topK,  # Max amount of results
            with_payload=True,
            **kwargs)

        paperIds = [pt.payload['paperId'] for pt in results.points]
        chunk_condition = FieldCondition(
            key='paperId', match=models.MatchAny(any=paperIds))
        query_filter = Filter(must=chunk_condition)

        chunk_results = await self.client.query_points_groups(
            collection_name=self.collection_name_chunks,
            query=query_vector,
            query_filter=query_filter,
            group_by="paperIdQ",
            limit=topK,  # Max amount of groups
            group_size=topK_per_paper,  # Max amount of points per group
            with_payload=True,
            **kwargs)

        best_texts = {}
        for group in chunk_results.groups:
            paperId = group.hits[0].payload['paperId']
            best_texts[paperId] = [group.hits[k].payload['text']
                                   for k in range(len(group.hits))]
        for pt in results.points:
            pt.payload['best_chunks'] = best_texts[pt.payload['paperId']]

        return results

    async def query_batch_streamed(self,
                                   query_vectors: List[List[float]],
                                   query_ids: List[int] = None,
                                   topK: int = 5,
                                   topK_per_paper: int = 5,
                                   payload_filters: List[List] = None, **kwargs):
        if not query_ids:
            query_ids = [k for k in range(len(query_vectors))]
        if not payload_filters:
            payload_filters = [None for _ in range(len(query_vectors))]

        tasks = [
            self.query(query_vector=vec,
                       topK=topK,
                       topK_per_paper=topK_per_paper,
                       payload_filter=filter,
                       **kwargs)
            for vec, filter in zip(query_vectors, payload_filters)
        ]
        results = await asyncio.gather(*tasks)
        final_results = {qid: res for qid, res in zip(query_ids, results)}
        return final_results

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


def build_qdrant_filter(payload_filter: Optional[List[Tuple]] = None):
    """
    Build a Qdrant Filter object using provided structured payload filters.

    Each top-level filter is combined using AND (`must`), but filters like
    ['authorName', 'any', [...]] create OR clauses inside a must entry.
    """
    conditions = {'must': []}

    if payload_filter:
        matchany_fields = ['paperIdQ', 'paperId', 'venue', 'fieldsOfStudy']
        matchtext_fields = ['abstract', 'main_text', 'title', 'authorName']
        matchrange_fields = ['publicationYear',
                             'citationCount', 'influentialCitationCount']
        matchdate_fields = ['publicationDate']

        for filt in payload_filter:
            field, op, value = filt

            if field in matchany_fields:
                values = value if isinstance(value, list) else [value]
                if op in {'==', 'eq'}:
                    match = models.MatchAny(any=values)
                elif op == '!=':
                    match = models.MatchExcept(**{"except": values})
                else:
                    continue  # unsupported operator
                conditions['must'].append(
                    FieldCondition(key=field, match=match))

            elif field in matchtext_fields:
                texts = value if isinstance(value, list) else [value]
                sub_conditions = [FieldCondition(
                    key=field, match=models.MatchText(text=txt)) for txt in texts]
                # Wrap in a nested Filter to simulate OR (should) under AND (must)
                if op == 'any':
                    conditions['must'].append(
                        Filter(should=sub_conditions, must=[]))
                else:
                    # default to all (AND) if not "any"
                    conditions['must'].extend(sub_conditions)

            elif field in matchrange_fields:
                cond = {op: value}
                val_range = models.Range(**cond)
                conditions['must'].append(
                    FieldCondition(key=field, range=val_range))

            elif field in matchdate_fields:
                cond = {op: value}
                val_range = models.DatetimeRange(**cond)
                conditions['must'].append(
                    FieldCondition(key=field, range=val_range))

    return Filter(**conditions)
