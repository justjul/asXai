import re
from pathlib import Path
from multiprocessing import Process, Event
from datetime import datetime
from typing import Optional, List, Any

from dataset.download_and_extract_PDF import download_and_extract
from dataset.embed_and_push import embed_and_push

import logging
from src.logger import get_logger
from src.utils import load_params

logger = get_logger(__name__, level=logging.INFO)

params = load_params()
pdf_config = params["pdf"]


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
        keep_pdfs: Optional[bool] = pdf_config['keep_pdfs']):

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    push_to_vectorDB = True

    download_extract_proc = Process(
        target=download_and_extract,
        args=(years, filters, n_jobs,
              timeout_loadpage, timeout_startdw,
              save_pdfs_to, timeout_per_article,
              keep_pdfs, push_to_vectorDB))

    embed_push_proc = Process(
        target=embed_and_push,
        args=(None,))

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
        keep_pdfs: Optional[bool] = pdf_config['keep_pdfs']):

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    if download_extract and embed_push:
        extract_embed_and_push(years=years, filters=filters, n_jobs=n_jobs,
                               timeout_loadpage=timeout_loadpage, timeout_startdw=timeout_startdw,
                               save_pdfs_to=save_pdfs_to, timeout_per_article=timeout_per_article,
                               keep_pdfs=keep_pdfs)

    if download_extract and not embed_push:
        download_and_extract(years=years, filters=filters, n_jobs=n_jobs,
                             timeout_loadpage=timeout_loadpage, timeout_startdw=timeout_startdw,
                             save_pdfs_to=save_pdfs_to, timeout_per_article=timeout_per_article,
                             keep_pdfs=keep_pdfs)

    if embed_push and not download_extract:
        embed_and_push(years=years, filters=filters)
