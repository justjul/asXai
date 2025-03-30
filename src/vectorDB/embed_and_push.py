import re
from multiprocessing import Process, Event
import asyncio
import pandas as pd
from datetime import datetime
from typing import Optional

from .utils_qdrant import QdrantManager
from .utils_embeddings import PaperEmbed
from src.dataset.dataset_loader import load_dataset
from src.dataset.utils_PDF import get_clean_block_text

import logging
from src.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def run_qdrant(done_event):
    manager = QdrantManager()

    async def run_and_watch():
        await manager.sync_qdrant(done_event)

    asyncio.run(run_and_watch())


def run_embedding(
        done_event,
        years: Optional[int] = None):

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    producer = PaperEmbed()

    for year in years:
        paperdata = load_dataset(subsets=year,
                                 data_types=["text", "metadata"],
                                 filters=[('pdf_blocks', '!=', 'None'), ('abstract', '!=', 'None')])
        data = get_payload_and_text(paperdata)

        logger.info(f"Will now embed {len(data)} articles for year {year}")
        producer.batch_embeddings(data.iloc)

    producer.unload_model()
    done_event.set()


def get_normalized_textdata(textdata):
    papertext = textdata.copy()
    papertext['pdf_blocks'] = papertext['pdf_blocks'].fillna('')
    papertext['pdf_blocks'] = papertext['pdf_blocks'].replace(to_replace='None',
                                                              value='')
    papertext['pdf_extracted'] = papertext['pdf_blocks'].str.strip().str.len() > 5

    papertext['pdf_blocks'] = papertext['pdf_blocks'].apply(
        lambda x: get_clean_block_text(x))

    mask = papertext["pdf_blocks"].str.len() < 5
    papertext.loc[mask, "pdf_blocks"] = papertext.loc[mask].apply(
        lambda x: ' '.join([x["title"], x["abstract"]]), axis=1)

    papertext = papertext.rename(columns={"pdf_blocks": 'text'})
    papertext = papertext[['paperId', 'text', 'pdf_extracted', 'title',
                           'abstract',]]
    return papertext


def get_normalized_metadata(metadata):
    return metadata


def get_payload_and_text(paperdata):
    textdata = get_normalized_textdata(paperdata["text"])
    metadata = get_normalized_metadata(paperdata["metadata"])
    data = pd.merge(metadata, textdata, how='left', on='paperId')

    return data


def embed_and_push(years: Optional[int] = None):

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    done_event = Event()

    sync_proc = Process(target=run_qdrant, args=(done_event,))
    embed_proc = Process(target=run_embedding,
                         args=(done_event, years))

    sync_proc.start()
    embed_proc.start()

    embed_proc.join()
    sync_proc.join()
