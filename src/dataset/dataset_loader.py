import multiprocessing.pool
import config
import os
from typing import List, Tuple, Union, TypedDict
from datetime import datetime
import pandas as pd

import logging
from src.logger import get_logger

import multiprocessing
from functools import partial

logger = get_logger(__name__, level=logging.INFO)


class paperData(TypedDict):
    metadata: pd.DataFrame
    text: pd.DataFrame


class DataSetManager:
    def __init__(
            self,
            dataset_source: str = "s2",
            data_types: Union[str, List[str]] = ["metadata", "text"],
            filters: Union[Tuple] | List[List[Tuple]] = None):

        self.dataset_source = dataset_source
        self.data_types = data_types
        self.filters = filters

    def _load_subset(
            self,
            subset: Union[int, str, List[str], List[int]] = None) -> paperData:

        data_path = {
            "metadata": config.METADATA_PATH,
            "text": config.TEXTDATA_PATH}
        paperdata = {
            "metadata": [],
            "text": []}

        paperIds = None
        for i, dtype in enumerate(self.data_types):
            path = data_path.get(dtype, None)
            if not path:
                continue

            file_name = f"{self.dataset_source}_{dtype}_{subset}.parquet"
            file_path = path / subset / file_name

            if i == 0:
                _filters = self.filters
            else:
                _filters = [('paperId', 'in', paperIds)]

            # fallback for missing pdf parquet
            if not file_path.exists():
                file_name = f"{self.dataset_source}_{dtype}0_{subset}.parquet"
                file_path = path / subset / file_name

            if file_path.exists():
                df = pd.read_parquet(
                    file_path, engine="pyarrow", filters=_filters)
                paperdata[dtype] = df
                paperIds = df['paperId']
            else:
                logger.info(f"No {dtype} data for {subset}")

        return paperdata


def load_worker_init(
        dataset_source: str = "s2",
        subsets: int = datetime.now().year,
        data_types: Union[str, List[str]] = ["metadata", "text"],
        filters: Union[Tuple] | List[List[Tuple]] = None):
    global datasetmanager
    datasetmanager = DataSetManager(dataset_source,
                                    subsets=subsets,
                                    data_types=data_types,
                                    filters=filters)


def load_subset(subset):
    global datasetmanager
    output = datasetmanager._load_subset(subset)
    return output


def load_dataset(
        dataset_source: str = "s2",
        subsets: Union[int, str, List[str], List[int]] = None,
        data_types: Union[str, List[str]] = ["metadata", "text"],
        filters: Union[List[Tuple], List[List[Tuple]]] = None,
        save_as: str = None, n_jobs=1) -> paperData:
    """Fetch dataset metadata and text data for given years and data types."""
    if dataset_source not in config.DATA_SOURCES:
        raise NameError(f"data_source should be one of {config.DATA_SOURCES}")

    # Normalize inputs
    if subsets is None:
        subsets = [year for year in os.listdir(
            config.METADATA_PATH) if year.isdigit() and len(year) == 4]
    elif isinstance(subsets, int) or isinstance(subsets, str):
        subsets = [str(subsets)]
    else:
        subsets = [str(subsets) for subsets in subsets]

    data_types = data_types or ["metadata", "text"]
    data_types = [data_types] if isinstance(data_types, str) else data_types

    # Initialize loaders
    data_map = {
        "metadata": (config.METADATA_PATH, []),
        "text": (config.TEXTDATA_PATH, [])}

    load_pool = multiprocessing.Pool(processes=n_jobs,
                                     initializer=partial(load_worker_init,
                                                         dataset_source=dataset_source,
                                                         data_types=data_types,
                                                         filters=filters))
    with load_pool:
        results = load_pool.map_async(load_subset, subsets).get()
        for res in results:
            for dtype, data in res.items():
                data_map[dtype][1].append(data)

    # Concatenate and clean outputs
    output = {}
    if "metadata" in data_types:
        output["metadata"] = (pd.concat(data_map["metadata"][1], ignore_index=True)
                              if data_map["metadata"][1]
                              else pd.DataFrame())
    if "text" in data_types:
        output["text"] = (pd.concat(data_map.get("text")[1], ignore_index=True)
                          if data_map.get("text")[1]
                          else pd.DataFrame())

    if ("metadata" in data_types
        and "text" in data_types
            and len(output["metadata"]) != len(output["text"])):
        logger.error("Metadata and text data don't have the same length.")
        raise ValueError("Metadata and text data don't have the same length.")

    if save_as:
        for dtype in data_types:
            if not output[dtype].empty:
                dir_path = data_map[dtype][0] / save_as
                dir_path.mkdir(parents=True, exist_ok=True)

                file_name = f"{dataset_source}_{dtype}_{save_as}.parquet"
                file_path = dir_path / file_name

                output[dtype].to_parquet(file_path, engine="pyarrow",
                                         compression="snappy", index=False)

    return output
