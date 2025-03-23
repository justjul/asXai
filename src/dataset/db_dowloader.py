import config
import os
import time
from datetime import datetime
from tqdm import tqdm

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from .utils import merge_dicts

import requests
import logging
from src.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def s2_validate_fields(
        fields: Optional[List[str]],
        allowed: set) -> List[str]:

    if not fields:
        return list(allowed)
    return [field.strip() for field in fields if field in allowed]


def get_s2_dataset_batch(
        endpoint: str,
        token: Optional[Union[str, int]] = None,
        check_token: bool = False) -> dict:

    token_str = f'&token={token}' if token else ''
    waitsec = 1
    dataset_list = None
    while ((dataset_list is None or 'total' not in dataset_list or
           (dataset_list.get('token') is None and check_token)) and waitsec < 100):
        dataset_list = requests.get(endpoint + token_str).json()
        if 'total' not in dataset_list or (dataset_list.get('token') is None and check_token):
            logger.info("Failed to download batch. Will try again...")
            print(dataset_list)
        time.sleep(waitsec)
        waitsec += 1
    return dataset_list


def get_s2_articles_batch(
        endpoint: str,
        papers: pd.DataFrame,
        specs: str) -> pd.DataFrame:

    minibatch_size = 500
    paper_specs = []
    for k in range(0, len(papers), minibatch_size):
        batch_ids = papers.paperId.iloc[k:k+minibatch_size].to_list()
        waitsec = 1
        r = None
        while (r is None or not isinstance(r, list)) and waitsec < 100:
            try:
                r = requests.post(endpoint, params={'fields': specs},
                                  json={"ids": batch_ids}).json()
            except requests.exceptions.JSONDecodeError:
                logger.error("Invalid JSON response received. Trying again")
            time.sleep(waitsec)
            print(waitsec, '\n', r)
            waitsec += 1
        if r:
            df_temp = pd.DataFrame([doc for doc in r if doc is not None])
            if not df_temp.empty:
                df_temp['openAccessPdf'] = df_temp['openAccessPdf'].apply(
                    lambda x: x['url'] if isinstance(x, dict) else x)

                df_temp['referenceIds'] = df_temp['references'].apply(
                    lambda x: ';'.join([ref['paperId'] for ref in x if ref.get('paperId')]))

                df_temp['referenceTitles'] = df_temp['references'].apply(lambda x: ';'.join(
                    [ref['title'].replace(';', ',') for ref in x if ref.get('title')]))

                df_temp.drop(columns=['references'], inplace=True)
                df_temp['influentialCitationCount'] = df_temp['influentialCitationCount'].fillna(
                    0).astype(float)

                paper_specs.append(df_temp)

    return pd.concat(paper_specs, ignore_index=True) if paper_specs else pd.DataFrame()


def s2_to_dataframe(dataset_list: dict) -> pd.DataFrame:

    papers = pd.json_normalize(dataset_list['data'])
    authors = pd.json_normalize(papers.pop('authors').apply(merge_dicts))
    papers = pd.concat([papers, authors], axis=1)
    papers = papers.dropna(subset=['publicationDate', 'name', 'fieldsOfStudy'])
    papers = papers.drop_duplicates(subset='title')
    papers = papers[papers['venue'].str.len() > 3]
    papers['authorId'] = papers['authorId'].apply(
        lambda x: ','.join(filter(None, x)))

    papers['authorName'] = papers['name'].apply(lambda x: ','.join(x))

    papers['fieldsOfStudy'] = papers['fieldsOfStudy'].apply(
        lambda x: ','.join(filter(None, x)))

    papers = papers.drop(columns='name')
    return papers


def get_s2_articles_year(
        endpoint: str,
        additional_specs: str,
        year: int) -> pd.DataFrame:

    dataset_list = get_s2_dataset_batch(endpoint)
    N_articles = dataset_list['total']
    token = 0
    articles = []
    batch_size = len(dataset_list['data'])

    with tqdm(range(0, N_articles, batch_size), desc=f'{year} ({N_articles} papers)') as pbar:
        for start_idx in pbar:
            dataset_list = get_s2_dataset_batch(
                endpoint, token=token, check_token=(start_idx < N_articles - batch_size))
            token = dataset_list.get('token')
            df = s2_to_dataframe(dataset_list)
            df['publicationYear'] = year

            pbar.set_postfix({'batch start': start_idx + 1, 'token': token})

            if additional_specs:
                spec_endpoint = 'https://api.semanticscholar.org/graph/v1/paper/batch'
                df_specs = get_s2_articles_batch(
                    spec_endpoint, df, additional_specs)
                df = df.merge(df_specs, on='paperId', how='left')
                df['influentialCitationCount'] = df['influentialCitationCount'].fillna(
                    0)
                df['openAccessPdf'] = df['openAccessPdf'].fillna('None')

            articles.append(df)
            pbar.set_postfix({'batch start': start_idx + 1, 'token': token,
                              'status': f"{len(df)} new articles"})
            if not token:
                if start_idx < N_articles - batch_size:
                    pbar.close()
                    logger.error(
                        "Returned token is None before end of dataset")
                    raise ValueError(
                        "Returned token is None before end of dataset")
                break

    df = pd.concat(articles)
    df = df.drop_duplicates(subset='paperId')
    df = df.reset_index(drop=True)
    df = df.apply(lambda x: x.astype(str).str.strip().str.encode(
        'utf-8', 'ignore').str.decode('utf-8') if x.dtype == object else x)
    logger.info(f'Total articles for {year}: {len(df)}')
    return df


def s2_db_download(
        years: Union[int, List[int]] = datetime.now().year,
        min_citations_per_year: float = 1,
        fields_of_study: Optional[List[str]] = ['Computer Science', 'Biology'],
        fields_to_return: Optional[List[str]] = [
            'title', 'citationCount', 'abstract', 'venue', 'authors', 'publicationDate', 'fieldsOfStudy'],
        additional_specs: Optional[List[str]] = [
            'influentialCitationCount', 'openAccessPdf', 'references'],
        overwrite: bool = False):

    years = [years] if isinstance(years, int) else years
    citation_thresholds = np.maximum(
        0, min_citations_per_year * ((datetime.now().year - 2) - np.array(years)))

    fields_of_study_str = ','.join(s2_validate_fields(fields_of_study, {
        "Computer Science", "Biology", "Medicine", "Physics", "Geology",
        "Psychology", "Mathematics", "Environmental Science", "Agricultural and Food Sciences"}))

    fields_to_return_str = ','.join(s2_validate_fields(fields_to_return, {
        'title', 'citationCount', 'abstract', 'venue', 'authors', 'publicationDate', 'fieldsOfStudy'}))

    specs_str = ','.join(s2_validate_fields(additional_specs, {
        "influentialCitationCount", "openAccessPdf", "references", "authors.affiliations", "references.paperId", "embedding.specter_v2"}))

    for year, min_citation in zip(years, citation_thresholds):
        endpoint = (
            f'https://api.semanticscholar.org/graph/v1/paper/search/bulk'
            f'?fields={fields_to_return_str}&fieldsOfStudy={fields_of_study_str}'
            f'&minCitationCount={int(min_citation)}&year={year}&openAccessPdf'
        )
        articles = get_s2_articles_year(endpoint, specs_str, year)

        year_metadata_dir = os.path.join(config.METADATA_PATH, str(year))
        year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))
        os.makedirs(year_metadata_dir, exist_ok=True)
        os.makedirs(year_text_dir, exist_ok=True)

        metadata_fp = os.path.join(
            year_metadata_dir, f's2_metadata_{year}.parquet')
        text_fp = os.path.join(year_text_dir, f's2_text0_{year}.parquet')

        metadata = articles.drop(columns=['title', 'abstract'])
        text = articles[['paperId', 'title', 'abstract',
                         'referenceTitles', 'openAccessPdf']]

        if not overwrite:
            if os.path.isfile(metadata_fp):
                old_metadata = pd.read_parquet(metadata_fp)
                metadata = metadata.set_index("paperId").combine_first(
                    old_metadata.set_index("paperId")).reset_index()
            if os.path.isfile(text_fp):
                old_text = pd.read_parquet(text_fp)
                text = text.set_index("paperId").combine_first(
                    old_text.set_index("paperId")).reset_index()

        metadata.to_parquet(metadata_fp, engine="pyarrow",
                            compression="snappy", index=False)
        text.to_parquet(text_fp, engine="pyarrow",
                        compression="snappy", index=False)


def s2_db_update(
        years: Union[int, List[int]] = datetime.now().year,
        specs: Optional[List[str]] = [
            'citationCount', 'influentialCitationCount', 'openAccessPdf', 'references']):

    years = [years] if isinstance(years, int) else years

    specs_str = ','.join(s2_validate_fields(specs, {
        "citationCount", "influentialCitationCount", "openAccessPdf", "references", "authors.affiliations", "references.paperId", "embedding.specter_v2"}))

    for year in years:
        year_metadata_dir = os.path.join(config.METADATA_PATH, str(year))
        year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))

        metadata_fp = os.path.join(year_metadata_dir,
                                   f's2_metadata_{year}.parquet')
        text0_fp = os.path.join(year_text_dir, f's2_text0_{year}.parquet')
        text_fp = os.path.join(year_text_dir, f's2_text_{year}.parquet')

        old_metadata = pd.read_parquet(metadata_fp)
        old_text0 = pd.read_parquet(text0_fp)
        if os.path.isfile(text_fp):
            old_text = pd.read_parquet(text_fp)
        else:
            old_text = None

        spec_endpoint = 'https://api.semanticscholar.org/graph/v1/paper/batch'
        df_specs = get_s2_articles_batch(spec_endpoint,
                                         old_metadata, specs_str)
        metadata = df_specs[old_metadata.columns.intersection(
            df_specs.columns)]
        text0 = df_specs[old_text0.columns.intersection(df_specs.columns)]

        metadata = metadata.set_index("paperId").combine_first(
            old_metadata.set_index("paperId")).reset_index()

        metadata.to_parquet(metadata_fp, engine="pyarrow",
                            compression="snappy", index=False)

        text0 = text0.set_index("paperId").combine_first(
            old_text0.set_index("paperId")).reset_index()

        text0.to_parquet(text0_fp, engine="pyarrow",
                         compression="snappy", index=False)

        if old_text:
            text = text0.set_index("paperId").combine_first(
                old_text.set_index("paperId")).reset_index()
            text.to_parquet(text_fp, engine="pyarrow",
                            compression="snappy", index=False)
