import os
import glob
import config
import docker
import docker.errors
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, Match

from typing import List, Tuple, Union, TypedDict
from datetime import datetime
from pathlib import Path
import json
import time
from tqdm import tqdm

import logging
from src.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class QdrantManager:
    def __init__(self,
                 collection_name: str = config.VECTOR_SIZE.keys()[0],
                 docker_timeout: float = 5.0,
                 watch_tmp_timeout: float = 60.0):
        self.volume_path = config.VECTORDB_PATH
        self.tmp_path = config.VECTORDB_PATH / "tmp"
        self.container_name = config.PROJECT_NAME + "-Qdrant"
        self.collection_name = collection_name
        self.vector_size = config.VECTOR_SIZE[collection_name]
        self.port = config.QDRANT_PORT
        self.host = config.QDRANT_HOST
        self.docker_timeout = docker_timeout
        self.watch_tmp_timeout = watch_tmp_timeout

        self.start_qdrant_container()

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
                                                  self.volume_path: {
                                                      'bind': 'qdrant/storage',
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

    def get_existing_paperIds(self):
        paperIds = set()
        scroll_cursor = None

        while True:
            scroll_result, scroll_cursor = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                with_payload=False,
                with_vectors=False,
                limit=1000,  # smaller batch size for pagination
                offset=scroll_cursor
            )
            for point in scroll_result:
                paperIds.add(point.id.split("_")[0])
            if scroll_cursor is None:
                break

        return paperIds

    def get_embedding_ids(self):
        files = list(Path(self.tmp_path).glob("*.json"))
        return [f.stem for f in files]

    async def delete_papers(self, paperIds):
        delete_tasks = [
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="paperId",
                                                            match=Match(value=id))])
            )
            for id in paperIds
        ]
        await asyncio.gather(*delete_tasks)

    async def upload_paper(self, paperId):
        file_path = self.tmp_path / f"{paperId}.json"
        with open(file_path, "r") as f:
            data = json.load(f)
        embeddings = data['embeddings']
        payloads = data['payloads']
        points = [PointStruct(id=f"{paperId}_{k}", vector=emb, payload=payloads[k])
                  for k, emb in enumerate(embeddings)]
        await self.client.upsert(collection_name=self.collection_name, points=points)
        os.remove(file_path)

    async def sync_qdrant(self):
        self.client = AsyncQdrantClient(host=self.host, port=self.port)

        if not await self.client.collection_exists(self.collection_name):
            await self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size,
                                            distance=Distance.COSINE))

        qdrant_paperIds = self.get_existing_paperIds(self)

        endtime = time.time() + self.watch_tmp_timeout
        paper_ids = []
        while paper_ids or time.time() < endtime:
            paper_ids = self.get_embedding_ids()

            if paper_ids:
                await self.delete_papers(paper_ids)

                upload_tasks = [self.upload_paper(id) for id in paper_ids]
                with tqdm(asyncio.as_completed(upload_tasks),
                          total=len(upload_tasks),
                          desc='Uploading') as pbar:
                    for f in pbar:
                        await f
                endtime = time.time() + self.watch_tmp_timeout
