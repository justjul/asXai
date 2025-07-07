"""
asXai Dataset Download Module
-----------------------------

Handles downloading and processing of scientific papers from:
- Semantic Scholar API (bulk search & batch endpoints)
- arXiv Kaggle dataset

Provides functions to:
- Validate user-specified fields to return
- Fetch dataset metadata batches with retry logic
- Normalize JSON responses into pandas DataFrames
- Download and update per-year datasets
- Map arXiv categories to fields of study
- Synchronize arXiv PDF links
- Filter and merge arXiv data into existing local datasets
"""

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

# Initialize module-level logger
logger = get_logger(__name__, level=config.LOG_LEVEL)

# Load download-specific configuration
params = load_params()
s2_config = params["download"]


def s2_validate_fields(
        fields: Optional[List[str]],
        allowed: set
) -> List[str]:
    """
    Ensure requested fields intersect with allowed set, or return all allowed if none provided.

    Args:
        fields: List of field names the user wants, or None.
        allowed: Set of allowed field names.

    Returns:
        List of validated field names.
    """
    if not fields:
        return list(allowed)
    return [field.strip() for field in fields if field in allowed]


def get_s2_dataset_batch(
        endpoint: str,
        token: Optional[Union[str, int]] = None,
        check_token: bool = False
) -> dict:
    """
    Fetch one batch of the Semantic Scholar dataset search endpoint, with retry until valid.

    Args:
        endpoint: Full URL for the /paper/search/bulk endpoint.
        token: Continuation token if paginating.
        check_token: If True, ensure 'token' is present in response.

    Returns:
        JSON-decoded dict containing 'data', 'total', and optional 'token'.
    """
    token_str = f'&token={token}' if token else ''
    waitsec = 1
    dataset_list = None

    # Retry until we get 'total' and (if requested) a valid token
    while ((dataset_list is None or 'total' not in dataset_list or
           (dataset_list.get('token') is None and check_token)) and waitsec < 256):
        dataset_list = requests.get(endpoint + token_str).json()
        if 'total' not in dataset_list or (dataset_list.get('token') is None and check_token):
            logger.info("Failed to download batch. Will try again...")
        time.sleep(waitsec)
        waitsec *= 2  # exponential backoff
    return dataset_list


def get_s2_articles_batch(
        endpoint: str,
        papers: pd.DataFrame,
        specs: str
) -> pd.DataFrame:
    """
    Fetch additional article fields (e.g., references, openAccessPdf) in batches of IDs.

    Args:
        endpoint: /paper/batch endpoint URL.
        papers: DataFrame with 'paperId' column to query.
        specs: Comma-separated string of fields to retrieve.

    Returns:
        DataFrame with additional specs merged across all requested IDs.
    """

    minibatch_size = 500
    paper_specs = []

    # Process in minibatches to avoid too-large JSON bodies
    for k in range(0, len(papers), minibatch_size):
        batch_ids = papers.paperId.iloc[k:k+minibatch_size].to_list()
        waitsec = 1
        r = None

        # Retry until we get a list of results
        while (r is None or not isinstance(r, list)) and waitsec < 256:
            try:
                r = requests.post(
                    endpoint,
                    params={'fields': specs},
                    json={"ids": batch_ids}
                ).json()
            except requests.exceptions.JSONDecodeError:
                logger.error(
                    f"Invalid JSON response received. Will try again in {waitsec}")
            time.sleep(waitsec)
            waitsec *= 2
        if r:
            df_temp = pd.DataFrame([doc for doc in r if doc is not None])
            if not df_temp.empty:
                # Clean up nested fields
                df_temp['references'] = df_temp['references'].fillna('')

                df_temp['openAccessPdf'] = df_temp['openAccessPdf'].apply(
                    lambda x: x['url'] if isinstance(x, dict) else x)

                df_temp['doi'] = df_temp['externalIds'].apply(
                    lambda x: x.get('DOI', '') or x.get('ArXiv', '') or x.get('CorpusId', '') if isinstance(x, dict) else 'None')

                # Flatten reference IDs and titles
                df_temp['referenceIds'] = df_temp['references'].apply(
                    lambda x: ';'.join([ref['paperId'] for ref in (x or []) if ref.get('paperId')]))

                df_temp['referenceTitles'] = df_temp['references'].apply(lambda x: ';'.join(
                    [ref['title'].replace(';', ',') for ref in (x or []) if ref.get('title')]))

                df_temp.drop(columns=['references'], inplace=True)
                df_temp['influentialCitationCount'] = df_temp['influentialCitationCount'].fillna(
                    0).astype(float)

                paper_specs.append(df_temp)

    return pd.concat(paper_specs, ignore_index=True) if paper_specs else pd.DataFrame()


def s2_to_dataframe(dataset_list: dict) -> pd.DataFrame:
    """
    Normalize the Semantic Scholar search response into a flat DataFrame.

    Args:
        dataset_list: JSON dict from get_s2_dataset_batch containing 'data'.

    Returns:
        DataFrame with selected columns and normalized author fields.
    """
    # Flatten top-level fields
    papers = pd.json_normalize(dataset_list['data'])
    # Normalize nested authors
    authors = pd.json_normalize(papers.pop('authors').apply(merge_dicts))
    papers = pd.concat([papers, authors], axis=1)
    # papers = papers.dropna(subset=['publicationDate', 'name', 'fieldsOfStudy'])

    # Drop incomplete or duplicate entries
    papers = papers.dropna(subset=['name'])
    papers = papers.drop_duplicates(subset='title')
    # Keep only plausible venues
    papers = papers[papers['venue'].str.len() > 3]
    papers["venue_lower"] = papers['venue'].astype(str).str.lower()

    # Convert list fields to comma-separated strings
    papers['authorId'] = papers['authorId'].apply(
        lambda x: ','.join(str(e) for e in x if e) if isinstance(x, Iterable) else '')

    papers['authorName'] = papers['name'].apply(
        lambda x: ','.join(str(e) for e in x if e) if isinstance(x, Iterable) else '')

    papers['fieldsOfStudy'] = papers['fieldsOfStudy'].apply(
        lambda x: ','.join(str(e) for e in x if e) if isinstance(x, Iterable) else '')

    # Final cleanup and column selection
    papers['citationCount'] = papers['citationCount'].fillna(0).astype(float)

    papers = papers.drop(columns='name')

    papers = papers[['paperId', 'title', 'abstract', 'venue', 'venue_lower', 'citationCount',
                     'fieldsOfStudy', 'publicationDate', 'authorId', 'authorName']]
    return papers


def get_s2_articles_year(
        endpoint: str,
        additional_specs: str,
        year: int
) -> pd.DataFrame:
    """
    Download all articles for a given year using Semantic Scholar APIs.

    Args:
        endpoint: Bulk search endpoint with query parameters embedded.
        additional_specs: Comma-separated extra fields for batch endpoint.
        year: Publication year to fetch.

    Returns:
        Consolidated DataFrame of articles for that year.
    """
    # Initial batch to get total count
    dataset_list = get_s2_dataset_batch(endpoint)
    N_articles = dataset_list['total']
    token = 0
    articles = []
    batch_size = len(dataset_list['data'])

    with tqdm(range(0, N_articles, batch_size), desc=f'{year} ({N_articles} papers)') as pbar:
        for start_idx in pbar:
            # Fetch each paginated batch
            dataset_list = get_s2_dataset_batch(
                endpoint, token=token, check_token=(start_idx < N_articles - batch_size))
            token = dataset_list.get('token')
            df = s2_to_dataframe(dataset_list)
            df['publicationYear'] = year
            df['source'] = 's2'
            df['status'] = None

            pbar.set_postfix({'batch start': start_idx + 1, 'token': token})

            # Fetch and merge additional fields if requested
            if additional_specs:
                spec_endpoint = 'https://api.semanticscholar.org/graph/v1/paper/batch'
                df_specs = get_s2_articles_batch(
                    spec_endpoint, df, additional_specs)

                df = df.merge(df_specs, on='paperId',
                              how='left', suffixes=('', '_spec'))

                # Drop spec duplicates and fill missing numeric fields
                df = df.drop(
                    columns=[col for col in df.columns if col.endswith('_spec')])
                df['influentialCitationCount'] = df['influentialCitationCount'].fillna(
                    0)
                df['openAccessPdf'] = df['openAccessPdf'].fillna('None')

                # Keep only articles that are either openAccess or have at least
                # their abstract available
                df = df[df['isOpenAccess'] | (
                    df['abstract'].notnull() & df['abstract'].astype(bool))]

            articles.append(df)
            pbar.set_postfix({
                'batch start': start_idx + 1,
                'status': f"{len(df)} new articles",
                'token': token,
            })
            if not token:
                if start_idx < N_articles - batch_size:
                    pbar.close()
                    logger.error(
                        "Returned token is None before end of dataset")
                    raise ValueError(
                        "Returned token is None before end of dataset")
                break

    df = pd.concat(articles)
    df = df.drop_duplicates(subset='doi')
    df = df.reset_index(drop=True)

    # Clean string columns
    df = df.apply(lambda x: x.where(x.isna(),
                                    x.astype(str).str.strip().str.encode(
                                        'utf-8', 'ignore').str.decode('utf-8'))
                  if x.dtype == object else x)
    # Turn back isOpenAccess as bool
    df['isOpenAccess'] = df['isOpenAccess'].astype(
        str).str.lower().map({'true': True, 'false': False})
    df['citationCount'] = df['citationCount'].astype(int)
    df['influentialCitationCount'] = df['influentialCitationCount'].astype(int)

    logger.info(f'Total articles for {year}: {len(df)}')
    return df


def update(
    years: Union[int, List[int]] = None,
    min_citations_per_year: float = None,
    fields_of_study: Optional[List[str]] = None,
    fields_to_return: Optional[List[str]] = None,
) -> None:
    """
    Main entrypoint: download and store per-year metadata and text parquet files,
    then trigger arXiv updates.

    Args:
        years: Single year or list of years (default: current year).
        min_citations_per_year: Minimum citations threshold.
        fields_of_study: List of fields to include.
        fields_to_return: List of metadata fields to fetch.
    """
    # Set defaults from config
    if years is None:
        years = datetime.now().year
    if min_citations_per_year is None:
        min_citations_per_year = s2_config.get("min_citations_per_year", 0)
    if fields_of_study is None:
        fields_of_study = s2_config.get("fields_of_study", [])
    if fields_to_return is None:
        fields_to_return = s2_config.get("fields_to_return", [])

    years = [years] if isinstance(years, int) else years
    # Compute citation thresholds per year
    citation_thresholds = np.maximum(
        0, min_citations_per_year * ((datetime.now().year - 2) - np.array(years)))
    citation_thresholds = np.minimum(20, citation_thresholds)

    # Validate and join field lists
    fields_of_study_str = ','.join(s2_validate_fields(fields_of_study, {
        "Computer Science", "Biology", "Medicine", "Physics", "Geology",
        "Psychology", "Mathematics", "Environmental Science", "Agricultural and Food Sciences"}))

    fields_to_return_str = ','.join(s2_validate_fields(fields_to_return, {
        'title', 'citationCount', 'abstract', 'venue', 'authors', 'publicationDate', 'fieldsOfStudy'}))

    specs_str = ','.join(s2_validate_fields(fields_to_return, {
        "influentialCitationCount", "openAccessPdf", "isOpenAccess", "references", "externalIds", "authors.affiliations", "references.paperId", "embedding.specter_v2"}))

    logger.info(f"will now load papers for {years} \n"
                f"fields of study: {fields_of_study_str}\n"
                f"returning fields: {fields_to_return_str + specs_str}")

    for year, min_citation in zip(years, citation_thresholds):
        # Prepare directories and in-progress flags
        year_metadata_dir = config.METADATA_PATH / str(year)
        year_text_dir = config.TEXTDATA_PATH / str(year)
        os.makedirs(year_metadata_dir, exist_ok=True)
        os.makedirs(year_text_dir, exist_ok=True)

        # Creating dummy files to signal update is in progress
        meta_inprogress_dummy = year_metadata_dir / "inprogress.pkl"
        with open(meta_inprogress_dummy, "wb") as f:
            pickle.dump([0], f, protocol=pickle.HIGHEST_PROTOCOL)

        txt_inprogress_dummy = year_text_dir / "inprogress.pkl"
        with open(txt_inprogress_dummy, "wb") as f:
            pickle.dump([0], f, protocol=pickle.HIGHEST_PROTOCOL)

        # endpoint = (
        #     f'https://api.semanticscholar.org/graph/v1/paper/search/bulk'
        #     f'?fields={fields_to_return_str}&fieldsOfStudy={fields_of_study_str}'
        #     f'&minCitationCount={int(min_citation)}&year={year}&openAccessPdf'
        # )

        endpoint = (
            f'https://api.semanticscholar.org/graph/v1/paper/search/bulk'
            f'?fields={fields_to_return_str}&fieldsOfStudy={fields_of_study_str}'
            f'&minCitationCount={int(min_citation)}&year={year}'
        )

        articles = get_s2_articles_year(endpoint, specs_str, year)

        # Split into metadata vs text
        metadata = articles.drop(columns=['title', 'abstract', 'status'])
        text = articles[['paperId', 'title', 'abstract',
                         'referenceTitles', 'openAccessPdf', 'isOpenAccess', 'status', 'doi']]
        text0 = text.copy()

        # Merge with existing files if they exist
        for df, filename in [(metadata, year_metadata_dir / f"metadata_{year}.parquet"),
                             (text0, year_text_dir / f"text0_{year}.parquet"),
                             (text, year_text_dir / f"text_{year}.parquet")]:
            if filename.exists():
                old = pd.read_parquet(filename)
                old = old.replace('None', None)
                df = df.set_index("doi").combine_first(
                    old.set_index("doi")).reset_index()
            # Replacing missing string values
            string_cols = df.select_dtypes(
                include=["object", "string"]).columns
            df.loc[:, string_cols] = df[string_cols].fillna("None")
            df.to_parquet(filename, engine="pyarrow",
                          compression="snappy", index=False)

        # if os.path.isfile(metadata_fp):
        #     old_metadata = pd.read_parquet(metadata_fp)
        #     old_metadata = old_metadata.replace('None', None)
        #     metadata = metadata.set_index("doi").combine_first(
        #         old_metadata.set_index("doi")).reset_index()
        # if os.path.isfile(text0_fp):
        #     old_text0 = pd.read_parquet(text0_fp)
        #     old_text0 = old_text0.replace('None', None)
        #     text0 = text.set_index("doi").combine_first(
        #         old_text0.set_index("doi")).reset_index()
        # if os.path.isfile(text_fp):
        #     old_text = pd.read_parquet(text_fp)
        #     old_text = old_text.replace('None', None)
        #     text = text.set_index("doi").combine_first(
        #         old_text.set_index("doi")).reset_index()

        # # drop previous incorporation of arxiv papers that have now been
        # # incorporated into s2.
        # dupli_mask = text0.set_index("doi").duplicated(
        #     subset="title", keep=False)
        # arxiv_mask = metadata.set_index("doi")['authorId'].isna()
        # drop_mask = dupli_mask & arxiv_mask
        # metadata = metadata.set_index("doi")[~drop_mask].reset_index()
        # text0 = text0.set_index("doi")[~drop_mask].reset_index()
        # text = text.set_index("doi")[~drop_mask].reset_index()

        # metadata = metadata.fillna('None')
        # metadata.to_parquet(metadata_fp, engine="pyarrow",
        #                     compression="snappy", index=False)
        # text0 = text0.fillna('None')
        # text0.to_parquet(text0_fp, engine="pyarrow",
        #                  compression="snappy", index=False)
        # text = text.fillna('None')
        # text.to_parquet(text_fp, engine="pyarrow",
        #                 compression="snappy", index=False)

        # Trigger arXiv updates and link fixes
        arX_update(years=year)

        # Remove in-progress flags
        os.remove(meta_inprogress_dummy)
        os.remove(txt_inprogress_dummy)


# def s2_db_update(
#         years: Union[int, List[int]] = datetime.now().year,
#         specs: Optional[List[str]] = [
#             'citationCount', 'influentialCitationCount', 'openAccessPdf', 'references']):

#     years = [years] if isinstance(years, int) else years

#     specs_str = ','.join(s2_validate_fields(specs, {
#         "citationCount", "influentialCitationCount", "openAccessPdf", "references", "authors.affiliations", "references.paperId", "embedding.specter_v2"}))

#     for year in years:
#         year_metadata_dir = os.path.join(config.METADATA_PATH, str(year))
#         year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))

#         metadata_fp = os.path.join(year_metadata_dir,
#                                    f'metadata_{year}.parquet')
#         text0_fp = os.path.join(year_text_dir, f'text0_{year}.parquet')
#         text_fp = os.path.join(year_text_dir, f'text_{year}.parquet')

#         old_metadata = pd.read_parquet(metadata_fp)
#         old_text0 = pd.read_parquet(text0_fp)
#         if os.path.isfile(text_fp):
#             old_text = pd.read_parquet(text_fp)
#         else:
#             old_text = None

#         spec_endpoint = 'https://api.semanticscholar.org/graph/v1/paper/batch'
#         df_specs = get_s2_articles_batch(spec_endpoint,
#                                          old_metadata, specs_str)
#         metadata = df_specs[old_metadata.columns.intersection(
#             df_specs.columns)]
#         text0 = df_specs[old_text0.columns.intersection(df_specs.columns)]

#         metadata = metadata.set_index("doi").combine_first(
#             old_metadata.set_index("doi")).reset_index()

#         metadata.to_parquet(metadata_fp, engine="pyarrow",
#                             compression="snappy", index=False)

#         text0 = text0.set_index("doi").combine_first(
#             old_text0.set_index("doi")).reset_index()

#         text0.to_parquet(text0_fp, engine="pyarrow",
#                          compression="snappy", index=False)

#         if old_text:
#             text = text0.set_index("doi").combine_first(
#                 old_text.set_index("doi")).reset_index()
#             text.to_parquet(text_fp, engine="pyarrow",
#                             compression="snappy", index=False)


# Mapping of arXiv categories to human-readable fields
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


def map_arxiv_category(categories) -> str:
    """
    Convert space-separated arXiv category codes into a comma-separated list of fields.

    Args:
        categories: Space-separated arXiv category strings (e.g. "cs.AI cs.LG").

    Returns:
        Comma-separated string of mapped fields, or 'Other' if unmapped.
    """
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
    """
    Normalize the arXiv Kaggle metadata snapshot into our schema.

    Args:
        arX_data: Raw DataFrame loaded from arxiv-metadata-oai-snapshot.json.

    Returns:
        Transformed DataFrame with columns aligned to asXai pipeline.
    """
    arX_norm = arX_data.rename(columns={
        "id": "paperId", "authors": "authorName",
        "categories": "fieldsOfStudy"
    })
    # Build PDF URL, map fields, cleanup authors string
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
    arX_norm["venue_lower"] = 'arxiv.org'
    arX_norm["isOpenAccess"] = True
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
    arX_norm['status'] = None
    arX_norm['doi'] = arX_norm['doi'].fillna(
        "10.48550/arXiv." + arX_norm['paperId'])

    return arX_norm


def load_arX_dataset(arXiv_downloads_dir) -> pd.DataFrame:
    """
    Ensure the arXiv Kaggle snapshot is downloaded and loaded as a DataFrame.
    Caches snapshot to parquet and refreshes if older than one week.

    Args:
        download_dir: Directory to store the Kaggle dataset.

    Returns:
        DataFrame of the full arXiv metadata snapshot.
    """
    arXiv_df_path = os.path.join(
        arXiv_downloads_dir, "arxiv-metadata-oai-snapshot.parquet")

    if (os.path.isfile(arXiv_df_path)
            and time.time() - os.path.getmtime(arXiv_df_path) < 3600*24*7):
        return pd.read_parquet(arXiv_df_path)

    # Otherwise download via kaggle CLI
    logger.info(
        f"updating arXiv database from Kaggle to {arXiv_downloads_dir}...")
    subprocess.run([
        "kaggle", "datasets", "download", "Cornell-University/arxiv",
        "--path", str(arXiv_downloads_dir), "--unzip"
    ], check=True)

    data_path = os.path.join(
        arXiv_downloads_dir, "arxiv-metadata-oai-snapshot.json")
    arX_data = pd.read_json(data_path, lines=True)

    arX_data = arX_to_dataframe(arX_data)

    arX_data.to_parquet(arXiv_df_path, engine="pyarrow",
                        compression="snappy", index=False)

    return arX_data


def arXlinks_update(years: Union[int, List[int]] = datetime.now().year):
    """
    Replace PDF URLs in existing text0 files with arXiv URLs when available.

    Args:
        years: Single year or list of years to update.
    """
    arXiv_downloads_dir = config.TMP_PATH / "arXiv"
    os.makedirs(arXiv_downloads_dir, exist_ok=True)

    years = [years] if isinstance(years, int) else years
    arX_data = load_arX_dataset(arXiv_downloads_dir)

    for year in years:
        year_metadata_dir = os.path.join(config.METADATA_PATH, str(year))
        year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))

        # Load existing tables
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

        # Save back
        metadata.to_parquet(metadata_fp, engine="pyarrow",
                            compression="snappy", index=False)
        text0.to_parquet(text0_fp, engine="pyarrow",
                         compression="snappy", index=False)
        text.to_parquet(text_fp, engine="pyarrow",
                        compression="snappy", index=False)

        logger.info(f"arXiv links for year {year} updated")


def filter_arXiv_database(arX_data) -> pd.DataFrame:
    """
    Remove records from the arXiv DataFrame that already appear in local text0 titles.

    Args:
        arX_data: Full arXiv DataFrame.

    Returns:
        Filtered DataFrame without duplicates.
    """
    years = sorted(os.listdir(config.TEXTDATA_PATH))
    filt_arXiv = arX_data
    for year in years:
        year_text_dir = os.path.join(config.TEXTDATA_PATH, str(year))

        text0_fp = os.path.join(year_text_dir, f'text0_{year}.parquet')
        text0 = pd.read_parquet(text0_fp)

        filt_arXiv = filt_arXiv[~filt_arXiv['doi'].str.lower().isin(
            text0['title'].str.lower().to_list())]

    return filt_arXiv


def arX_update(years: Union[int, List[int]] = datetime.now().year) -> None:
    """
    Incorporate new arXiv papers into local metadata/text datasets by field-of-study.

    Args:
        years: Single year or list of years to update.
    """
    arXiv_downloads_dir = config.TMP_PATH / "arXiv"
    os.makedirs(arXiv_downloads_dir, exist_ok=True)

    years = [years] if isinstance(years, int) else years
    arX_data = load_arX_dataset(arXiv_downloads_dir)

    # arX_data = filter_arXiv_database(arX_data)

    for year in years:
        year_metadata_dir = config.METADATA_PATH / str(year)
        year_text_dir = config.TEXTDATA_PATH / str(year)

        metadata_fp = year_metadata_dir / f'metadata_{year}.parquet'

        if not os.path.isfile(metadata_fp):
            logger.info(
                f"No database found for year {year}: download articles from s2 first")
            continue

        logger.info(f"updating database with arXiv papers for year {year}")

        old_metadata = pd.read_parquet(metadata_fp)

        # Filter arXiv by fields-of-study present in old metadata
        fields = old_metadata['fieldsOfStudy'].apply(
            lambda x: x.split(',')[0]).unique()

        field_mask = arX_data['fieldsOfStudy'].apply(
            lambda x: any(c in fields for c in x.split(',')))

        # Prepare new metadata and text
        arX_data_new = arX_data[field_mask]

        # arX_data_new = arX_data_new[~arX_data_new['title'].isin(
        #     old_text0)]
        arX_data_new = arX_data_new[arX_data_new['publicationYear'].astype(
            int) == year]
        arX_data_new = arX_data_new.drop_duplicates(subset='title')
        arX_data_new = arX_data_new.drop_duplicates(subset='doi')

        arX_metadata = arX_data_new.drop(columns=['title', 'abstract', 'submitter',
                                                  'comments', 'journal-ref',
                                                  'report-no', 'license', 'versions',
                                                  'authors_parsed', 'status'])
        arX_text = arX_data_new[['paperId',
                                 'title', 'abstract',
                                 'openAccessPdf', 'isOpenAccess', 'status', 'doi']]

        # Merge with existing
        # Merge with existing files if they exist
        for df, filename in [(arX_metadata, year_metadata_dir / f"metadata_{year}.parquet"),
                             (arX_text, year_text_dir /
                              f"text0_{year}.parquet"),
                             (arX_text, year_text_dir / f"text_{year}.parquet")]:
            if filename.exists():
                old = pd.read_parquet(filename)
                old = old.replace('None', None)
                df = df.set_index("doi").combine_first(
                    old.set_index("doi")).reset_index()
            df['status'] = None
            # Filling missing citation values with 0
            if 'citationCount' in df.columns:
                df['citationCount'] = df['citationCount'].fillna(0).astype(int)
            if 'influentialCitationCount' in df.columns:
                df['influentialCitationCount'] = df['influentialCitationCount'].fillna(
                    0).astype(int)
            # Replacing missing string values
            string_cols = df.select_dtypes(
                include=["object", "string"]).columns
            df[string_cols] = df[string_cols].fillna("None")
            df.to_parquet(filename, engine="pyarrow",
                          compression="snappy", index=False)
            logger.info(
                f"added {len(df)} , {len(df) - len(old)}")

        logger.info(f"updating arXiv links for year {year}")
        arXlinks_update(years=year)
