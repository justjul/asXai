from pathlib import Path
import os
import shutil
import time
import asyncio
import pandas as pd

from multiprocessing import Process, Event
from datetime import datetime
from typing import Optional, Union, List, Tuple, Any

from asxai.pdf.extract_PDF import extract_PDFs
from asxai.pdf.download_PDF import download_PDFs

from asxai.vectorDB.qdrant import QdrantManager
from asxai.vectorDB.embed import PaperEmbed
from asxai.dataIO.load import load_data, update_text
from asxai.pdf.extract_PDF import collect_extracted_batch

import config
from asxai.logger import get_logger
from asxai.utils import load_params

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
pdf_config = params["pdf"]


def download_and_extract(**kwargs):

    for year in kwargs['years']:
        paperdata = load_data(subsets=year,
                              data_types=["metadata", "text"])
        paperdata_filt = paperdata
        if kwargs['filters'] is not None:
            paperdata_filt = load_data(subsets=year,
                                       data_types=[
                                           "text", "metadata"],
                                       filters=kwargs['filters'])
        download_proc = Process(
            target=download_PDFs,
            kwargs={'paperdata': paperdata_filt,
                    'year': year,
                    'n_jobs': kwargs['n_jobs'][0],
                    'timeout_loadpage': kwargs['timeout_loadpage'],
                    'timeout_startdw': kwargs['timeout_startdw'],
                    'save_pdfs_to': kwargs['save_pdfs_to']})

        extract_proc = Process(
            target=extract_PDFs,
            kwargs={'paperdata': paperdata_filt,
                    'year': year,
                    'n_jobs': kwargs['n_jobs'][1],
                    'timeout_per_article': kwargs['timeout_per_article'],
                    'max_pages': kwargs['max_pages'],
                    'pdfs_dir': kwargs['save_pdfs_to'],
                    'keep_pdfs': kwargs['keep_pdfs'],
                    'push_to_vectorDB': kwargs.get('push_to_vectorDB', False),
                    'extract_done': kwargs.get('extract_done', None)})

        extract_proc.start()
        download_proc.start()

        download_proc.join()
        extract_proc.join()

        tmp_dir_year = config.TMP_PATH / "text_to_save" / str(year)
        extracted_path = os.path.join(tmp_dir_year, f"text_{year}.parquet")
        final_textdata = pd.read_parquet(extracted_path, engine="pyarrow")
        update_text(final_textdata, year)
        shutil.rmtree(tmp_dir_year)


def run_embedding(
        done_event,
        years: Optional[Union[int, List[int]]] = None,
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
            config.TMP_PATH, "text_to_embed"))
        os.makedirs(extracted_dir, exist_ok=True)
        os.chmod(extracted_dir, 0o777)
        while True:
            extracted_batches = collect_extracted_batch(extracted_dir)

            if extract_done.is_set() and not extracted_batches:
                break

            if extracted_batches:
                paperdata = {'metadata': [],
                             'text': []}
                for batch_dir in extracted_batches:
                    paper_dir = extracted_dir / batch_dir

                    fp_metadata = paper_dir / "metadata.extracted"
                    paperdata['metadata'].append(
                        pd.read_parquet(fp_metadata, engine="pyarrow"))
                    if os.path.isfile(fp_metadata):
                        os.remove(fp_metadata)

                    fp_text = paper_dir / "text.extracted"
                    paperdata['text'].append(
                        pd.read_parquet(fp_text, engine="pyarrow"))
                    if os.path.isfile(fp_text):
                        os.remove(fp_text)

                    if os.path.isdir(paper_dir):
                        os.rmdir(paper_dir)

                paperdata['metadata'] = pd.concat(
                    paperdata['metadata'], axis=0).reset_index(drop=True)
                paperdata['text'] = pd.concat(
                    paperdata['text'], axis=0).reset_index(drop=True)

                logger.info(
                    f"Will now embed {len(paperdata['metadata'])} articles that just got extracted")
                producer.batch_embeddings(paperdata)

                # extracted_batches = collect_extracted_batch(extracted_dir)
            else:
                time.sleep(5)

        if os.path.isdir(extracted_dir):
            os.removedirs(extracted_dir)

    producer.unload_model()
    done_event.set()


def sync_qdrant(done_event):
    manager = QdrantManager()

    async def watch_and_sync():
        await manager.sync_qdrant(done_event)

    asyncio.run(watch_and_sync())


def update_qdrant_payloads(years: Union[int, List[int]] = None):
    years = [years] if not isinstance(years, list) else years

    manager = QdrantManager()

    async def batch_update_payloads():
        for year in years:
            logger.info(f"Updating payloads for year {year}")
            paperdata = load_data(subsets=year,
                                  data_types=["text", "metadata"],
                                  filters=[('pdf_status', '==', 'extracted')]
                                  )
            metadata = paperdata['metadata'].to_dict(orient="records")
            await manager.batch_update_payloads(metadata)

    asyncio.run(batch_update_payloads())


def update_payloads(years: Union[int, List[int]] = None):
    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    payload_proc = Process(target=update_qdrant_payloads, args=(years,))

    payload_proc.start()
    payload_proc.join()


def embed_and_push(extract_done: Optional[bool] = None,
                   years: Optional[Union[int, List[int]]] = None,
                   filters: Optional[Union[List[Tuple],
                                           List[List[Tuple]]]] = None):

    embed_done = Event()

    sync_proc = Process(target=sync_qdrant, args=(embed_done,))
    embed_proc = Process(target=run_embedding,
                         args=(embed_done, years, filters, extract_done))

    sync_proc.start()
    embed_proc.start()

    embed_proc.join()
    sync_proc.join()


def extract_embed_and_push(
        years: Optional[Union[int, List[int]]] = None,
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
        years: Optional[Union[int, List[int]]] = None,
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
