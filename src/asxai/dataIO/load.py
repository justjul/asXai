"""
asXai Data I/O Module
---------------------

This module provides functionality to load, filter, and manage the source datasets
(metadata and text) used by the asXai RAG pipeline. It supports parallel loading of yearly subsets
of data, optional filtering, and saving/updating cleaned subsets back to disk.

Key Components:
- DataSetManager: Loads and filters per-subset parquet files.
- load_data: High-level entrypoint to fetch and optionally save combined datasets.
- update_text: Utility to merge new text data into existing parquet files.
"""

import config
import os
from typing import List, Tuple, Union, TypedDict
import time
import pandas as pd

from asxai.logger import get_logger
from asxai.utils import load_params

import multiprocessing
from functools import partial

logger = get_logger(__name__, level=config.LOG_LEVEL)

# Load parameters from config
params = load_params()
load_config = params["load"]


class paperData(TypedDict):
    """TypedDict representing the loaded data for one subset."""
    metadata: pd.DataFrame
    text: pd.DataFrame


class DataSetManager:
    """
        Manager for loading and filtering parquet datasets for a given subset.

        Attributes:
            data_types: List of data types to load ("metadata" and/or "text").
            filters: Optional list of filter tuples to apply to the first dataset loaded.
    """

    def __init__(
            self,
            data_types: Union[str, List[str]] = ["metadata", "text"],
            filters: Union[Tuple] | List[List[Tuple]] = None):

        self.data_types = data_types
        self.filters = filters

    def _load_subset(
            self,
            subset: Union[int, str, List[str], List[int]] = None) -> paperData:
        """
        Load metadata and/or text data for a specific subset (e.g., a year).
        Args:
            subset: Identifier of the subset (year, string or integer).
        Returns:
            A paperData dict containing DataFrames for each requested data type.
        """

        # Base paths for metadata and text
        data_path = {
            "metadata": config.METADATA_PATH,
            "text": config.TEXTDATA_PATH}
        # Initialize empty DataFrames
        paperdata = {
            "metadata": pd.DataFrame(),
            "text": pd.DataFrame()}

        paperIds = None  # to chain filters for text based on loaded metadata
        for i, dtype in enumerate(self.data_types):
            path = data_path.get(dtype, None)
            if not path:
                continue

            # Build expected file path
            file_name = f"{dtype}_{subset}.parquet"
            file_path = path / subset / file_name

            # Wait if a concurrent update is in progress
            while True:
                update_inprogress = path / subset / "inprogress.pkl"
                if os.path.exists(update_inprogress):
                    time.sleep(5)
                else:
                    break

            # Define filters: first dtype uses provided filters, subsequent use paperId chaining
            _filters = self.filters if i == 0 else [
                ('paperId', 'in', paperIds)]

            # Fallback if file missing: try zero-padded prefix (originally sourced from s2)
            if not file_path.exists():
                file_name = f"{dtype}0_{subset}.parquet"
                file_path = path / subset / file_name

            # Load if exists, else log and continue
            if file_path.exists():
                df = pd.read_parquet(
                    file_path, engine="pyarrow", filters=_filters)
                paperdata[dtype] = df
                # Capture paper IDs for chaining to next data type
                paperIds = df['paperId']
            else:
                logger.info(f"No {dtype} data for {subset}")

        return paperdata


def load_worker_init(
        data_types: Union[str, List[str]] = ["metadata", "text"],
        filters: Union[Tuple] | List[List[Tuple]] = None):
    """
    Worker initializer for multiprocessing pool. Sets up a global DataSetManager instance.

    Args:
        data_types: Types of data to load in this worker.
        filters: Filters to apply when loading the first data type.
    """
    global datasetmanager
    datasetmanager = DataSetManager(data_types=data_types,
                                    filters=filters)


def load_subset(subset):
    """
    Entrypoint for each multiprocessing task: delegates to the global DataSetManager.

    Args:
        subset: The subset identifier passed by the pool.

    Returns:
        paperData dict from DataSetManager._load_subset.
    """
    global datasetmanager
    output = datasetmanager._load_subset(subset)
    return output


def load_data(
        subsets: Union[int, str, List[str], List[int]] = None,
        data_types: Union[str, List[str]] = ["metadata", "text"],
        filters: Union[List[Tuple], List[List[Tuple]]] = None,
        save_as: str = None,
        n_jobs: int = load_config['n_jobs']) -> paperData:
    """
    High-level function to load and optionally save combined datasets.

    Args:
        subsets: One or more subset identifiers (years) to load. If None, auto-detects by listing directories.
        data_types: Data types to load ("metadata", "text").
        filters: Filters to apply to the metadata load.
        save_as: If provided, saves the concatenated DataFrames under this subdirectory.
        n_jobs: Number of parallel processes to use.

    Returns:
        A paperData dict containing loaded (and concatenated) DataFrames.
    """
    # Normalize subset list
    if subsets is None:
        # Auto-detect subset directories that look like years
        subsets = [year for year in os.listdir(
            config.METADATA_PATH) if year.isdigit() and len(year) == 4]
    elif isinstance(subsets, int) or isinstance(subsets, str):
        subsets = [str(subsets)]
    else:
        subsets = [str(subset) for subset in subsets]

    # Cap number of jobs to CPU capacity
    n_jobs = min(n_jobs, len(subsets))
    n_jobs = min(n_jobs, 2 * multiprocessing.cpu_count() // 3)

    # Ensure data_types is a list
    data_types = data_types or ["metadata", "text"]
    data_types = [data_types] if isinstance(data_types, str) else data_types

    # Prepare structure to collect loaded pieces
    data_map = {
        "metadata": (config.METADATA_PATH, []),
        "text": (config.TEXTDATA_PATH, [])}

    # Launch multiprocessing pool
    load_pool = multiprocessing.Pool(processes=n_jobs,
                                     initializer=partial(load_worker_init,
                                                         data_types=data_types,
                                                         filters=filters))
    with load_pool:
        results = load_pool.map_async(load_subset, subsets).get()
        # Collect results
        for res in results:
            for dtype, data in res.items():
                data_map[dtype][1].append(data)

    # Concatenate and validate
    output = {}
    if "metadata" in data_types:
        output["metadata"] = (pd.concat(data_map["metadata"][1], ignore_index=True)
                              if data_map["metadata"][1]
                              else pd.DataFrame())
    if "text" in data_types:
        output["text"] = (pd.concat(data_map.get("text")[1], ignore_index=True)
                          if data_map.get("text")[1]
                          else pd.DataFrame())

    # Ensure matching lengths if both present
    if ("metadata" in data_types
        and "text" in data_types
            and len(output["metadata"]) != len(output["text"])):
        msg = "Metadata and text data don't have the same length."
        logger.error(msg)
        raise ValueError(msg)

    # Optionally save the concatenated DataFrames
    if save_as:
        for dtype in data_types:
            if not output[dtype].empty:
                dir_path = data_map[dtype][0] / save_as
                dir_path.mkdir(parents=True, exist_ok=True)

                file_name = f"{dtype}_{save_as}.parquet"
                file_path = dir_path / file_name

                output[dtype].to_parquet(file_path, engine="pyarrow",
                                         compression="snappy", index=False)

    return output


def update_text(newtxtdata, subset):
    """
    Merge new text data with existing text parquet for a given subset.

    Args:
        newtxtdata: DataFrame containing new text data with 'doi' column.
        subset: The subset (year) to update.

    Behavior:
        - Loads existing text data for subset.
        - Merges new rows by DOI, preferring new data.
        - Writes updated DataFrame back to parquet.
    """
    # Load existing text data
    txtdata = load_data(subset, data_types=['text'])

    # Combine on DOI, giving precedence to newtxtdata
    txtdata = (newtxtdata.set_index("doi").combine_first(
        txtdata["text"].set_index("doi")).reset_index(drop=False))

    # Write back
    filename = f"text_{str(subset)}.parquet"
    filepath = config.TEXTDATA_PATH / str(subset) / filename
    txtdata.to_parquet(filepath, engine="pyarrow",
                       compression="snappy", index=True)
