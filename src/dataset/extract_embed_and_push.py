from pathlib import Path
import os
import time
import asyncio
import pandas as pd

from multiprocessing import Process, Event
from datetime import datetime
from typing import Optional, Union, List, Tuple, Any

from pdf.extract_PDF import extract_PDFs
from pdf.download_PDF import download_PDFs

from vectorDB.push_qdrant import QdrantManager
from vectorDB.embed import PaperEmbed
from dataIO.load import load_data
from pdf.extract_PDF import collect_extracted_batch

import config
import logging
from src.logger import get_logger
from src.utils import load_params

logger = get_logger(__name__, level=logging.INFO)

params = load_params()
pdf_config = params["pdf"]


def download_and_extract(**kwargs):

    done_event = Event()

    download_proc = Process(
        target=download_PDFs,
        kwargs={'years': kwargs['years'],
                'filters': kwargs['filters'],
                'n_jobs': kwargs['n_jobs'][0],
                'timeout_loadpage': kwargs['timeout_loadpage'],
                'timeout_startdw': kwargs['timeout_startdw'],
                'save_pdfs_to': kwargs['save_pdfs_to'],
                'done_event': done_event})

    extract_proc = Process(
        target=extract_PDFs,
        kwargs={'years': kwargs['years'],
                'filters': kwargs['filters'],
                'n_jobs': kwargs['n_jobs'][1],
                'timeout_per_article': kwargs['timeout_per_article'],
                'max_pages': kwargs['max_pages'],
                'pdfs_dir': kwargs['save_pdfs_to'],
                'keep_pdfs': kwargs['keep_pdfs'],
                'push_to_vectorDB': kwargs.get('push_to_vectorDB', False),
                'done_event': done_event,
                'extract_done': kwargs.get('extract_done', None)})

    extract_proc.start()
    download_proc.start()

    download_proc.join()
    extract_proc.join()


def run_embedding(
        done_event,
        years: Optional[int] = None,
        filters: Optional[Union[List[Tuple], List[List[Tuple]]]] = None,
        extract_done: Optional[bool] = None):

    producer = PaperEmbed()

    if years is not None:
        years = [years] if not isinstance(years, list) else years
        for year in years:
            paperdata = load_data(subsets=year,
                                  data_types=["text", "metadata"],
                                  filters=filters)

            logger.info(
                f"Will now embed {len(paperdata['metadata'])} articles for year {year}")
            producer.batch_embeddings(paperdata)
    else:
        extracted_dir = Path(os.path.join(
            config.VECTORDB_PATH, "tmp", "extracted"))
        os.makedirs(extracted_dir, exist_ok=True)
        while True:
            extracted_batches = collect_extracted_batch(extracted_dir)

            if extract_done.is_set() and not extracted_batches:
                break

            if extracted_batches:
                paper_dir = extracted_dir / extracted_batches[0]
                paperdata = {'metadata': pd.DataFrame(),
                             'text': pd.DataFrame()}

                fp_metadata = paper_dir / "metadata.extracted"
                paperdata['metadata'] = pd.read_parquet(
                    fp_metadata, engine="pyarrow")
                os.remove(fp_metadata)

                fp_text = paper_dir / "text.extracted"
                paperdata['text'] = pd.read_parquet(fp_text, engine="pyarrow")
                os.remove(fp_text)

                os.rmdir(paper_dir)

                logger.info(
                    f"Will now embed {len(paperdata['metadata'])} articles that just got extracted")
                producer.batch_embeddings(paperdata)

                extracted_batches = collect_extracted_batch(extracted_dir)
            else:
                time.sleep(5)

        if os.path.isdir(extracted_dir):
            os.removedirs(extracted_dir)

    producer.unload_model()
    done_event.set()


def run_qdrant(done_event):
    manager = QdrantManager()

    async def run_and_watch():
        await manager.sync_qdrant(done_event)

    asyncio.run(run_and_watch())


def embed_and_push(extract_done: Optional[bool] = None,
                   years: Optional[int] = None,
                   filters: Optional[Union[List[Tuple],
                                           List[List[Tuple]]]] = None):

    embed_done = Event()

    sync_proc = Process(target=run_qdrant, args=(embed_done,))
    embed_proc = Process(target=run_embedding,
                         args=(embed_done, years, filters, extract_done))

    sync_proc.start()
    embed_proc.start()

    embed_proc.join()
    sync_proc.join()


def extract_embed_and_push(
        years: Optional[int] = None,
        filters: Optional[List[Any] | List[List[Any]]] = None,
        n_jobs: Optional[List[int]] = [
            pdf_config['n_jobs_download'], pdf_config['n_jobs_extract'],
        ],
        timeout_loadpage: Optional[float] = pdf_config['timeout_loadpage'],
        timeout_startdw: Optional[float] = pdf_config['timeout_startdw'],
        save_pdfs_to: Optional[str | Path] = pdf_config['save_pdfs_to'],
        timeout_per_article: Optional[float] = pdf_config['timeout_per_article'],
        max_pages: Optional[Union[int, List[int]]] = [
            pdf_config['max_pages_start'], pdf_config['max_pages_end']],
        keep_pdfs: Optional[bool] = pdf_config['keep_pdfs']):

    push_to_vectorDB = True
    extract_done = Event()

    download_extract_proc = Process(
        target=download_and_extract,
        kwargs={'years': years,
                'filters': filters,
                'n_jobs': n_jobs,
                'timeout_loadpage': timeout_loadpage,
                'timeout_startdw': timeout_startdw,
                'save_pdfs_to': save_pdfs_to,
                'timeout_per_article': timeout_per_article,
                'max_pages': max_pages,
                'keep_pdfs': keep_pdfs,
                'push_to_vectorDB': push_to_vectorDB,
                'extract_done': extract_done
                })

    embed_push_proc = Process(
        target=embed_and_push,
        args=(extract_done,))

    embed_push_proc.start()
    download_extract_proc.start()

    download_extract_proc.join()
    embed_push_proc.join()


def process(
        download_extract: bool = True,
        embed_push: bool = True,
        years: Optional[int] = None,
        filters: Optional[List[Any] | List[List[Any]]] = None,
        n_jobs: Optional[List[int]] = [
            pdf_config['n_jobs_download'], pdf_config['n_jobs_extract'],
        ],
        timeout_loadpage: Optional[float] = pdf_config['timeout_loadpage'],
        timeout_startdw: Optional[float] = pdf_config['timeout_startdw'],
        save_pdfs_to: Optional[str | Path] = pdf_config['save_pdfs_to'],
        timeout_per_article: Optional[float] = pdf_config['timeout_per_article'],
        max_pages: Optional[Union[int, List[int]]] = [
            pdf_config['max_pages_start'], pdf_config['max_pages_end']],
        keep_pdfs: Optional[bool] = pdf_config['keep_pdfs']):

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    if download_extract and embed_push:
        extract_embed_and_push(years=years, filters=filters, n_jobs=n_jobs,
                               timeout_loadpage=timeout_loadpage, timeout_startdw=timeout_startdw,
                               save_pdfs_to=save_pdfs_to, timeout_per_article=timeout_per_article,
                               max_pages=max_pages, keep_pdfs=keep_pdfs)

    if download_extract and not embed_push:
        download_and_extract(years=years, filters=filters, n_jobs=n_jobs,
                             timeout_loadpage=timeout_loadpage, timeout_startdw=timeout_startdw,
                             save_pdfs_to=save_pdfs_to, timeout_per_article=timeout_per_article,
                             max_pages=max_pages, keep_pdfs=keep_pdfs)

    if embed_push and not download_extract:
        embed_and_push(years=years, filters=filters)
