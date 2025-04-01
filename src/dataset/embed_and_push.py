import os
from pathlib import Path
import time
from multiprocessing import Process, Event
import asyncio
import pandas as pd
from datetime import datetime
from typing import Optional, Union, List, Tuple

from vectorDB.push_qdrant import QdrantManager
from vectorDB.embed import PaperEmbed
from dataIO.load import load_data
from pdf.extract_PDF import collect_extracted_batch

import config
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
        years: Optional[int] = None,
        filters: Optional[Union[List[Tuple], List[List[Tuple]]]] = None,):

    producer = PaperEmbed()

    if years is not None:
        years = [years] if not isinstance(years, list) else years
        for year in years:
            paperdata = load_data(subsets=year,
                                  data_types=["text", "metadata"],
                                  filters=filters)
            data = get_payload_and_text(paperdata)

            logger.info(f"Will now embed {len(data)} articles for year {year}")
            producer.batch_embeddings(data)
    else:
        extracted_dir = Path(os.path.join(config.VECTORDB_PATH, "extracted"))
        os.makedirs(extracted_dir, exist_ok=True)
        endtime = time.time() + 300
        while True:
            extracted_batches = collect_extracted_batch(extracted_dir)

            if time.time() >= endtime and not extracted_batches:
                break

            if extracted_batches:
                paperdata = {'metadata': pd.DataFrame(),
                             'text': pd.DataFrame()}

                fp_metadata = extracted_dir / \
                    extracted_batches[0] / "metadata.extracted"
                paperdata['metadata'] = pd.read_parquet(
                    fp_metadata, engine="pyarrow")
                os.remove(fp_metadata)

                fp_text = extracted_dir / \
                    extracted_batches[0] / "text.extracted"
                paperdata['text'] = pd.read_parquet(fp_text, engine="pyarrow")
                os.remove(fp_text)

                data = get_payload_and_text(paperdata)
                logger.info(
                    f"Will now embed {len(data)} articles that just got extracted")
                producer.batch_embeddings(data)

                extracted_batches = collect_extracted_batch(extracted_dir)
                endtime = time.time() + 300
            else:
                time.sleep(5)

        os.removedirs(extracted_dir)

    producer.unload_model()
    done_event.set()


def get_normalized_textdata(textdata):
    papertext = textdata.copy()
    papertext['full_text'] = papertext['full_text'].fillna('')
    papertext['full_text'] = papertext['full_text'].replace(to_replace='None',
                                                            value='')
    papertext['full_text'] = papertext['full_text'].str.strip().str.len() > 5

    mask = papertext["full_text"].str.len() < 5
    papertext.loc[mask, "full_text"] = papertext.loc[mask].apply(
        lambda x: ' '.join([x["title"], x["abstract"]]), axis=1)

    papertext = papertext[['paperId', 'full_text', 'pdf_extracted', 'title',
                           'abstract',]]
    return papertext


def get_normalized_metadata(metadata):
    return metadata


def get_payload_and_text(paperdata):
    textdata = get_normalized_textdata(paperdata["text"])
    metadata = get_normalized_metadata(paperdata["metadata"])
    data = pd.merge(metadata, textdata, how='left', on='paperId')

    return data


def embed_and_push(years: Optional[int] = None,
                   filters: Optional[Union[List[Tuple], List[List[Tuple]]]] = None):

    done_event = Event()

    sync_proc = Process(target=run_qdrant, args=(done_event,))
    embed_proc = Process(target=run_embedding,
                         args=(done_event, years, filters))

    sync_proc.start()
    embed_proc.start()

    embed_proc.join()
    sync_proc.join()
