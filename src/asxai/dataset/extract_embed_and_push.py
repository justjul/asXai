"""
asXai PDF Extraction, Embedding & Push Module
---------------------------------------------

Orchestrates the full pipeline of:
1. Downloading PDFs for selected years/filters.
2. Extracting text from PDFs in parallel.
3. Generating embeddings for each paper chunk.
4. Pushing data and payload updates to Qdrant vector database.

Key functions:
- download_and_extract: parallel PDF download + text extraction, then update text parquet.
- run_embedding: batch embeddings of extracted text, streaming to Qdrant.
- sync_qdrant: watches for embedding completion events to trigger Qdrant sync.
- update_qdrant_payloads: refreshes metadata payloads in Qdrant.
- embed_and_push: coordinates embedding + Qdrant sync in separate processes.
- extract_embed_and_push: end-to-end orchestration of download/extract/embed/push.
- process: entrypoint to run subsets of pipeline.
"""

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
from asxai.dataIO.load import load_data, update_text, update_status
from asxai.pdf.extract_PDF import collect_extracted_batch

import config
from asxai.logger import get_logger
from asxai.utils import load_params, load_parquet_dataset

# Initialize logger
logger = get_logger(__name__, level=config.LOG_LEVEL)

# Load module-specific config (e.g., parallelism, timeouts)
params = load_params()
pdf_config = params["pdf"]


def download_and_extract(**kwargs):
    """
    Download PDFs and extract text in parallel per year.

    Args:
        years: List of publication years to process.
        filters: Optional filters to select subset of papers.
        n_jobs: Tuple (#download_workers, #extract_workers).
        timeout_loadpage: Timeout for page load during download.
        timeout_startdw: Timeout for download start.
        save_pdfs_to: Directory to store downloaded PDFs.
        timeout_per_article: Max extraction time per article.
        max_pages: Tuple (start_pages, end_pages) to extract.
        keep_pdfs: If False, delete PDFs after extraction.
        push_to_vectorDB: If True, extractor flags will push directly to vector DB.
        extract_done: Event to signal extraction completion.
    """
    for year in kwargs['years']:
        # Load metadata + text (filtered or full)
        paperdata = load_data(subsets=year,
                              data_types=["metadata", "text"])
        paperdata_filt = paperdata
        if kwargs['filters'] is not None:
            paperdata_filt = load_data(subsets=year,
                                       data_types=[
                                           "text", "metadata"],
                                       filters=kwargs['filters'])

        # Spawn download and extract processes
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

        # Start extractor first so it can pick up files as soon as they arrive
        extract_proc.start()
        download_proc.start()
        download_proc.join()
        extract_proc.join()

        # Update local text parquet and clean up temp directory
        tmp_dir_year = config.TMP_PATH / "text_to_save" / str(year)
        extracted_path = tmp_dir_year / f"text_{year}"
        final_textdata = load_parquet_dataset(extracted_path)
        update_text(final_textdata, year)
        shutil.rmtree(tmp_dir_year)


def run_embedding(
        done_event,
        years: Optional[Union[int, List[int]]] = None,
        filters: Optional[Union[List[Tuple], List[List[Tuple]]]] = None,
        extract_done: Optional[bool] = None
):
    """
    Generate and push batch embeddings to Qdrant. Can run in two modes:
      1. Year-based: embed all text for specified years.
      2. Streaming: watch a directory for newly extracted batches.

    Args:
        done_event: Event to signal when embedding/sync should stop.
        years: Optionally, list of years to embed in batch mode.
        filters: Filters applied to load_data when embedding by year.
        extract_done: Event that signals when extraction is complete (stream mode).
    """

    producer = PaperEmbed()

    # Batch mode for specified years
    if years is not None:
        years = [years] if not isinstance(years, list) else years
        for year in years:
            paperdata = load_data(subsets=year,
                                  data_types=["text", "metadata"],
                                  filters=filters)

            logger.info(
                f"Will now embed {len(paperdata['metadata'])} articles for year {year}")
            producer.batch_embeddings(paperdata)

    # Stream mode: watch TMP_PATH/text_to_embed until extract_done + empty directory
    else:
        extracted_dir = Path(os.path.join(
            config.TMP_PATH, "text_to_embed"))
        os.makedirs(extracted_dir, exist_ok=True)
        os.chmod(extracted_dir, 0o777)
        while True:
            extracted_batches = collect_extracted_batch(extracted_dir)
            # Exit when extraction is done and no batches remain
            if extract_done.is_set() and not extracted_batches:
                break

            if extracted_batches:
                # Consolidate all batch DataFrames
                paperdata = {'metadata': [],
                             'text': []}
                for batch_dir in extracted_batches:
                    paper_dir = extracted_dir / batch_dir

                    fp_metadata = paper_dir / "metadata"
                    paperdata['metadata'].append(
                        load_parquet_dataset(fp_metadata))
                    if os.path.exist(fp_metadata):
                        shutil.rmtree(fp_metadata)

                    fp_text = paper_dir / "text"
                    paperdata['text'].append(
                        load_parquet_dataset(fp_text))
                    if os.path.exists(fp_text):
                        shutil.rmtree(fp_text)

                    # Clean up batch files
                    paper_dir.rmdir()

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

        # Remove empty watch directory
        if extracted_dir.exists():
            extracted_dir.rmdir()

    # Unload model resources and signal completion
    producer.unload_model()
    done_event.set()


def sync_qdrant(done_event):
    """
    Continuously sync local Qdrant operations until done_event is set.
    Runs asynchronously in its own process.
    """
    manager = QdrantManager()

    async def watch_and_sync():
        await manager.sync_qdrant(done_event)

    asyncio.run(watch_and_sync())


def update_qdrant_payloads(years: Union[int, List[int]] = None):
    """
    Update Qdrant payloads (metadata) for extracted-only papers by year.

    Args:
        years: Year or list of years to update.
    """
    years = [years] if not isinstance(years, list) else years

    manager = QdrantManager()

    async def batch_update_payloads():
        for year in years:
            logger.info(f"Updating payloads for year {year}")
            paperdata = load_data(subsets=year,
                                  data_types=["text", "metadata"],
                                  filters=[('status', '==', 'extracted')]
                                  )
            metadata = paperdata['metadata'].to_dict(orient="records")
            await manager.batch_update_payloads(metadata)

    asyncio.run(batch_update_payloads())


def update_payloads(years: Union[int, List[int]] = None):
    """
    Wrapper to launch payload updates in a separate process.

    Args:
        years: Year or list of years to update payloads for.
    """
    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    payload_proc = Process(target=update_qdrant_payloads, args=(years,))

    payload_proc.start()
    payload_proc.join()


def embed_and_push(
    extract_done: Optional[bool] = None,
        years: Optional[Union[int, List[int]]] = None,
        filters: Optional[Union[List[Tuple],
                                List[List[Tuple]]]] = None
):
    """
    Coordinate embedding and Qdrant sync in parallel processes.

    Args:
        extract_done: Event indicating extraction completion for streaming mode.
        years: Optional list of years to embed in batch mode.
        filters: Optional filters for load_data.
    """
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
    n_jobs: Optional[List[int]] = None,
    timeout_loadpage: Optional[float] = None,
    timeout_startdw: Optional[float] = None,
    save_pdfs_to: Optional[str | Path] = None,
    timeout_per_article: Optional[float] = None,
    max_pages: Optional[Union[int, List[int]]] = None,
    keep_pdfs: Optional[bool] = None,
):
    """
    Full end-to-end pipeline:
      1) download_and_extract
      2) embed_and_push

    Args:
        years: Years to process.
        filters: Filters for selecting papers.
        n_jobs: [download_workers, extract_workers].
        timeout_loadpage: Timeout for download page load.
        timeout_startdw: Timeout for download start.
        save_pdfs_to: Directory for PDF files.
        timeout_per_article: Timeout for per-article extraction.
        max_pages: [start_pages, end_pages].
        keep_pdfs: Whether to keep PDF files after extraction.
    """
    push_to_vectorDB = True

    # Prepare events
    extract_done = Event()

    # Default to config values if not provided
    n_jobs = n_jobs or [pdf_config['n_jobs_download'],
                        pdf_config['n_jobs_extract']]
    timeout_loadpage = timeout_loadpage or pdf_config['timeout_loadpage']
    timeout_startdw = timeout_startdw or pdf_config['timeout_startdw']
    save_pdfs_to = save_pdfs_to or pdf_config['save_pdfs_to']
    timeout_per_article = timeout_per_article or pdf_config['timeout_per_article']
    max_pages = max_pages or [
        pdf_config['max_pages_start'], pdf_config['max_pages_end']]
    keep_pdfs = keep_pdfs if keep_pdfs is not None else pdf_config['keep_pdfs']

    # Start download+extract and embed+push in parallel
    download_extract_proc = Process(
        target=download_and_extract,
        kwargs={
            'years': years,
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

    embed_push_proc = Process(target=embed_and_push, args=(extract_done,))

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
    keep_pdfs: Optional[bool] = pdf_config['keep_pdfs']
):
    """
    High-level entrypoint to run any combination of download/extract and embed/push.

    Args:
        download_extract: If True, run download_and_extract phase.
        embed_push: If True, run embed_and_push phase.
        (Other args passed through to extract_embed_and_push)
    """

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    if download_extract and embed_push:
        extract_embed_and_push(years=years, filters=filters, n_jobs=n_jobs,
                               timeout_loadpage=timeout_loadpage, timeout_startdw=timeout_startdw,
                               save_pdfs_to=save_pdfs_to, timeout_per_article=timeout_per_article,
                               max_pages=max_pages, keep_pdfs=keep_pdfs)
        update_status(status={'extracted': 'pdf_pushed'}, subsets=years)

    if download_extract and not embed_push:
        download_and_extract(years=years, filters=filters, n_jobs=n_jobs,
                             timeout_loadpage=timeout_loadpage, timeout_startdw=timeout_startdw,
                             save_pdfs_to=save_pdfs_to, timeout_per_article=timeout_per_article,
                             max_pages=max_pages, keep_pdfs=keep_pdfs)

    if embed_push and not download_extract:
        embed_and_push(years=years, filters=filters)
        update_status(
            status={'extracted': 'pdf_pushed',
                    'None': 'abstract_pushed'},
            subsets=years
        )
