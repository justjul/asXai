import subprocess
import config
import os
import time
from datetime import datetime
from tqdm import tqdm
from collections.abc import Iterable

from typing import List, Optional, Union
import numpy as np
import pandas as pd
import pickle
from asxai.utils import merge_dicts
from asxai.utils import load_params

import requests
from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
s2_config = params["download"]


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
                logger.error(
                    f"Invalid JSON response received. Will try again in {waitsec}")
            time.sleep(waitsec)
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
    # papers = papers.dropna(subset=['publicationDate', 'name', 'fieldsOfStudy'])
    papers = papers.dropna(subset=['name'])
    papers = papers.drop_duplicates(subset='title')
    papers = papers[papers['venue'].str.len() > 3]
    papers['authorId'] = papers['authorId'].apply(
        lambda x: ','.join(str(e) for e in x if e) if isinstance(x, Iterable) else '')

    papers['authorName'] = papers['name'].apply(
        lambda x: ','.join(str(e) for e in x if e) if isinstance(x, Iterable) else '')

    papers['fieldsOfStudy'] = papers['fieldsOfStudy'].apply(
        lambda x: ','.join(str(e) for e in x if e) if isinstance(x, Iterable) else '')

    papers = papers.drop(columns='name')

    papers = papers[['paperId', 'title', 'abstract', 'venue', 'citationCount',
                     'fieldsOfStudy', 'publicationDate', 'authorId', 'authorName']]
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
            df['source'] = 's2'
            df['pdf_status'] = None

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
    df = df.apply(lambda x: x.where(x.isna(),
                                    x.astype(str).str.strip().str.encode(
                                        'utf-8', 'ignore').str.decode('utf-8'))
                  if x.dtype == object else x)
    logger.info(f'Total articles for {year}: {len(df)}')
    return df


def update(
        years: Union[int, List[int]] = datetime.now().year,
        min_citations_per_year: float = s2_config['min_citations_per_year'],
        fields_of_study: Optional[List[str]] = s2_config['fields_of_study'],
        fields_to_return: Optional[List[str]] = s2_config['fields_to_return']):

    years = [years] if isinstance(years, int) else years
    citation_thresholds = np.maximum(
        0, min_citations_per_year * ((datetime.now().year - 2) - np.array(years)))

    fields_of_study_str = ','.join(s2_validate_fields(fields_of_study, {
        "Computer Science", "Biology", "Medicine", "Physics", "Geology",
        "Psychology", "Mathematics", "Environmental Science", "Agricultural and Food Sciences"}))

    fields_to_return_str = ','.join(s2_validate_fields(fields_to_return, {
        'title', 'citationCount', 'abstract', 'venue', 'authors', 'publicationDate', 'fieldsOfStudy'}))

    specs_str = ','.join(s2_validate_fields(fields_to_return, {
        "influentialCitationCount", "openAccessPdf", "references", "authors.affiliations", "references.paperId", "embedding.specter_v2"}))

    logger.info(f"will now load papers for {years} \n"
                f"fields of study: {fields_of_study_str}\n"
                f"returning fields: {fields_to_return_str}")

    for year, min_citation in zip(years, citation_thresholds):
        year_metadata_dir = os.path.join(config.METADATA_PATH, str(year))
        year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))
        os.makedirs(year_metadata_dir, exist_ok=True)
        os.makedirs(year_text_dir, exist_ok=True)

        # Creating dummy files to signal update is in progress
        meta_inprogress_dummy = os.path.join(year_metadata_dir,
                                             "inprogress.pkl")
        with open(meta_inprogress_dummy, "wb") as f:
            pickle.dump([0], f, protocol=pickle.HIGHEST_PROTOCOL)

        txt_inprogress_dummy = os.path.join(year_text_dir,
                                            "inprogress.pkl")
        with open(txt_inprogress_dummy, "wb") as f:
            pickle.dump([0], f, protocol=pickle.HIGHEST_PROTOCOL)

        endpoint = (
            f'https://api.semanticscholar.org/graph/v1/paper/search/bulk'
            f'?fields={fields_to_return_str}&fieldsOfStudy={fields_of_study_str}'
            f'&minCitationCount={int(min_citation)}&year={year}&openAccessPdf'
        )
        articles = get_s2_articles_year(endpoint, specs_str, year)

        metadata_fp = os.path.join(
            year_metadata_dir, f'metadata_{year}.parquet')
        text0_fp = os.path.join(year_text_dir, f'text0_{year}.parquet')
        text_fp = os.path.join(year_text_dir, f'text_{year}.parquet')

        metadata = articles.drop(columns=['title', 'abstract', 'pdf_status'])
        text = articles[['paperId', 'title', 'abstract',
                         'referenceTitles', 'openAccessPdf', 'pdf_status']]
        text0 = text.copy()

        if os.path.isfile(metadata_fp):
            old_metadata = pd.read_parquet(metadata_fp)
            old_metadata = old_metadata.replace('None', None)
            metadata = metadata.set_index("paperId").combine_first(
                old_metadata.set_index("paperId")).reset_index()
        if os.path.isfile(text0_fp):
            old_text0 = pd.read_parquet(text0_fp)
            old_text0 = old_text0.replace('None', None)
            text0 = text.set_index("paperId").combine_first(
                old_text0.set_index("paperId")).reset_index()
        if os.path.isfile(text_fp):
            old_text = pd.read_parquet(text_fp)
            old_text = old_text.replace('None', None)
            text = text.set_index("paperId").combine_first(
                old_text.set_index("paperId")).reset_index()

        # drop previous incorporation of arxiv papers that have now been
        # incorporated into s2.
        dupli_mask = text0.set_index("paperId").duplicated(
            subset="title", keep=False)
        arxiv_mask = metadata.set_index("paperId")['authorId'].isna()
        drop_mask = dupli_mask & arxiv_mask
        metadata = metadata.set_index("paperId")[~drop_mask].reset_index()
        text0 = text0.set_index("paperId")[~drop_mask].reset_index()
        text = text.set_index("paperId")[~drop_mask].reset_index()

        metadata = metadata.fillna('None')
        metadata.to_parquet(metadata_fp, engine="pyarrow",
                            compression="snappy", index=False)
        text0 = text0.fillna('None')
        text0.to_parquet(text0_fp, engine="pyarrow",
                         compression="snappy", index=False)
        text = text.fillna('None')
        text.to_parquet(text_fp, engine="pyarrow",
                        compression="snappy", index=False)

        arXlinks_update(years=year)

        # Removing dummy files to signal update is completed
        os.remove(meta_inprogress_dummy)
        os.remove(txt_inprogress_dummy)


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
                                   f'metadata_{year}.parquet')
        text0_fp = os.path.join(year_text_dir, f'text0_{year}.parquet')
        text_fp = os.path.join(year_text_dir, f'text_{year}.parquet')

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


arxiv_category_map = {
    'cs': 'Computer Science',
    'q-bio': 'Biology',
    'q-fin': 'Mathematics',
    'stat': 'Mathematics',
    'math': 'Mathematics',
    'physics': 'Physics',
    'astro-ph': 'Physics',
    'cond-mat': 'Physics',
    'gr-qc': 'Physics',
    'hep-ex': 'Physics',
    'hep-lat': 'Physics',
    'hep-ph': 'Physics',
    'hep-th': 'Physics',
    'nlin': 'Physics',
    'nucl-ex': 'Physics',
    'nucl-th': 'Physics',
    'quant-ph': 'Physics',
    'eess': 'Physics',
    'econ': 'Mathematics',
    'physics.geo-ph': 'Geology',
}


def map_arxiv_category(categories):
    mapped_fields = set()
    for category in categories.split():
        if category in arxiv_category_map:
            mapped_fields.add(arxiv_category_map[category])
        else:
            # Attempt to map based on prefixes
            prefix = category.split('.')[0]
            if prefix in arxiv_category_map:
                mapped_fields.add(arxiv_category_map[prefix])
            else:
                mapped_fields.add('Other')
    return ','.join(mapped_fields)


def arX_to_dataframe(arX_data: pd.DataFrame) -> pd.DataFrame:
    arX_norm = arX_data.rename(columns={"id": "paperId", "authors": "authorName",
                                        "categories": "fieldsOfStudy"})
    arX_norm["openAccessPdf"] = arX_norm.apply(lambda x: "gs://arxiv-dataset/arxiv/arxiv/pdf/"
                                               + x["paperId"].split('.')[0] + "/"
                                               + x["paperId"] +
                                               x['versions'][-1]['version']
                                               + ".pdf", axis=1)
    arX_norm["fieldsOfStudy"] = arX_norm["fieldsOfStudy"].apply(
        map_arxiv_category)
    arX_norm['authorName'] = arX_norm['authorName'].str.replace(
        r'\s+and\s+', ', ', regex=True)
    arX_norm["venue"] = 'arXiv.org'
    arX_norm["authorId"] = None
    arX_norm["citationCount"] = None
    arX_norm["influentialCitationCount"] = None
    arX_norm["publicationDate"] = arX_norm['versions'].apply(
        lambda x: x[0]['created'])
    arX_norm["publicationDate"] = pd.to_datetime(
        arX_norm["publicationDate"], format=None)
    arX_norm["publicationYear"] = arX_norm["publicationDate"].dt.year
    arX_norm["publicationDate"] = arX_norm["publicationDate"].dt.strftime(
        date_format="%Y-%m-%d")
    arX_norm = arX_norm.drop(columns=["update_date"])
    arX_norm['source'] = 'arXiv'
    arX_norm['pdf_status'] = None

    return arX_norm


def load_arX_dataset(arXiv_downloads_dir):
    arXiv_df_path = os.path.join(
        arXiv_downloads_dir, "arxiv-metadata-oai-snapshot.parquet")

    if (os.path.isfile(arXiv_df_path)
            and time.time() - os.path.getmtime(arXiv_df_path) < 3600*24*7):
        arX_data = pd.read_parquet(arXiv_df_path)
    else:
        logger.info(
            f"updating arXiv database from Kaggle to {arXiv_downloads_dir}...")
        subprocess.run(["kaggle", "datasets", "download", "Cornell-University/arxiv", "--path",
                        str(arXiv_downloads_dir), "--unzip"], check=True)

        data_path = os.path.join(
            arXiv_downloads_dir, "arxiv-metadata-oai-snapshot.json")
        arX_data = pd.read_json(data_path, lines=True)

        arX_data = arX_to_dataframe(arX_data)

        arX_data.to_parquet(arXiv_df_path, engine="pyarrow",
                            compression="snappy", index=False)

    return arX_data


def arXlinks_update(years: Union[int, List[int]] = datetime.now().year):

    arXiv_downloads_dir = config.TMP_PATH / "arXiv"
    os.makedirs(arXiv_downloads_dir, exist_ok=True)

    years = [years] if isinstance(years, int) else years
    arX_data = load_arX_dataset(arXiv_downloads_dir)

    for year in years:
        year_metadata_dir = os.path.join(config.METADATA_PATH, str(year))
        year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))

        metadata_fp = os.path.join(year_metadata_dir,
                                   f'metadata_{year}.parquet')
        text0_fp = os.path.join(year_text_dir, f'text0_{year}.parquet')
        text_fp = os.path.join(year_text_dir, f'text_{year}.parquet')

        metadata = pd.read_parquet(metadata_fp)
        text0 = pd.read_parquet(text0_fp)
        if os.path.isfile(text_fp):
            text = pd.read_parquet(text_fp)
        else:
            text = text0

        # Replacing http links with arXiv links where available
        openAccess_arXiv = arX_data[['openAccessPdf', 'paperId']]
        text0['arXiv_paperId'] = text0['openAccessPdf'].apply(
            lambda x: x.split('/')[-1])
        text0 = text0.merge(openAccess_arXiv[['paperId', 'openAccessPdf']], left_on='arXiv_paperId',
                            right_on='paperId', how='left', suffixes=(None, '_arxiv'))

        openAccessPdf = text0['openAccessPdf_arxiv'].combine_first(
            text0['openAccessPdf'])
        text0 = text0.drop(
            columns=['arXiv_paperId', 'paperId_arxiv', 'openAccessPdf_arxiv'])
        openAccessPdf[openAccessPdf.isna()]
        text0['openAccessPdf'] = openAccessPdf
        metadata['openAccessPdf'] = openAccessPdf
        text['openAccessPdf'] = openAccessPdf

        metadata.to_parquet(metadata_fp, engine="pyarrow",
                            compression="snappy", index=False)
        text0.to_parquet(text0_fp, engine="pyarrow",
                         compression="snappy", index=False)
        text.to_parquet(text_fp, engine="pyarrow",
                        compression="snappy", index=False)

        logger.info(f"updated arXiv links for year {year}")


def filter_arXiv_database(arX_data):
    years = sorted(os.listdir(config.TEXTDATA_PATH))
    filt_arXiv = arX_data
    for year in years:
        year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))

        text0_fp = os.path.join(year_text_dir, f'text0_{year}.parquet')
        text0 = pd.read_parquet(text0_fp)

        filt_arXiv = filt_arXiv[~filt_arXiv['title'].str.lower().isin(
            text0['title'].str.lower().to_list())]

    return filt_arXiv


def arX_update(years: Union[int, List[int]] = datetime.now().year):

    arXiv_downloads_dir = config.TMP_PATH / "arXiv"
    os.makedirs(arXiv_downloads_dir, exist_ok=True)

    years = [years] if isinstance(years, int) else years
    arX_data = load_arX_dataset(arXiv_downloads_dir)

    arX_data = filter_arXiv_database(arX_data)

    for year in years:
        year_metadata_dir = os.path.join(config.METADATA_PATH, str(year))
        year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))

        metadata_fp = os.path.join(year_metadata_dir,
                                   f'metadata_{year}.parquet')
        text0_fp = os.path.join(year_text_dir, f'text0_{year}.parquet')
        text_fp = os.path.join(year_text_dir, f'text_{year}.parquet')

        old_metadata = pd.read_parquet(metadata_fp)
        old_text0 = pd.read_parquet(text0_fp)
        if os.path.isfile(text_fp):
            old_text = pd.read_parquet(text_fp)
        else:
            old_text = old_text0

        fields = old_metadata['fieldsOfStudy'].apply(
            lambda x: x.split(',')[0]).unique()

        field_mask = arX_data['fieldsOfStudy'].apply(
            lambda x: any(c in fields for c in x.split(',')))
        arX_data_new = arX_data[field_mask]
        # arX_data_new = arX_data_new[~arX_data_new['title'].isin(
        #     old_text0)]
        arX_data_new = arX_data_new[arX_data_new['publicationYear'].astype(
            int) == year]
        arX_data_new = arX_data_new.drop_duplicates(subset='title')

        arX_metadata = arX_data_new.drop(columns=['title', 'abstract', 'submitter',
                                                  'comments', 'journal-ref', 'doi',
                                                  'report-no', 'license', 'versions',
                                                  'authors_parsed', 'pdf_status'])
        arX_text = arX_data_new[['paperId',
                                 'title', 'abstract',
                                 'openAccessPdf', 'pdf_status']]

        metadata = arX_metadata.set_index("paperId").combine_first(
            old_metadata.set_index("paperId")).reset_index()

        text0 = arX_text.set_index("paperId").combine_first(
            old_text0.set_index("paperId")).reset_index()

        text = arX_text.set_index("paperId").combine_first(
            old_text.set_index("paperId")).reset_index()

        # drop arxiv papers that have already been incorporated into s2.
        text0['title_lower'] = text0['title'].str.lower()
        dupli_mask = text0.set_index("paperId").duplicated(
            subset="title_lower", keep=False)
        text0 = text0.drop(columns=['title_lower'])

        arxiv_mask = metadata.set_index("paperId")['authorId'].isna()

        drop_mask = dupli_mask & arxiv_mask
        metadata = metadata.set_index("paperId")[~drop_mask].reset_index()
        text0 = text0.set_index("paperId")[~drop_mask].reset_index()
        text = text.set_index("paperId")[~drop_mask].reset_index()

        metadata.to_parquet(metadata_fp, engine="pyarrow",
                            compression="snappy", index=False)
        text0.to_parquet(text0_fp, engine="pyarrow",
                         compression="snappy", index=False)
        text.to_parquet(text_fp, engine="pyarrow",
                        compression="snappy", index=False)

        logger.info(
            f"added {len(arX_metadata)} , {len(metadata) - len(old_metadata)}")
        logger.info(f"added {len(arX_text)} , {len(text0) - len(old_text0)}")
        logger.info(f"added {len(arX_text)} , {len(text0) - len(old_text0)}")
