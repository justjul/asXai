import sys
import yaml
from pathlib import Path
from tqdm import tqdm as std_tqdm
from tqdm.notebook import tqdm as nb_tqdm
import os

import pyarrow.dataset as ds
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union, Optional, List, Tuple
import operator

import config

import asyncio
import threading

# Map string ops to Python operators
_OP_MAP = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "in": lambda field, val: field.isin(val),
    "not in": lambda field, val: ~field.isin(val),
}


class AsyncRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


def merge_dicts(dict_list):
    merged = {}
    for d in dict_list:
        for key, value in d.items():
            merged.setdefault(key, []).append(value)  # Collect values as lists
    return merged


def get_tqdm():
    if 'ipykernel' in sys.modules:
        return nb_tqdm  # Jupyter notebook or Lab
    else:
        return std_tqdm


def load_params(params_path: Path = Path(config.CODE_ROOT / "params.yaml")):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    params["qdrant"]["model_name"] = params["embedding"]["model_name"]

    return params


def load_parquet_dataset(
    path: Union[str, Path],
    filters: Optional[List[Tuple[str, str, object]]] = None
) -> pd.DataFrame:
    """
    Load a Parquet file or dataset folder with optional filters.

    Args:
        path: Path to a `.parquet` file or folder containing Parquet part files.
        filters: Optional pyarrow-style filters (list of tuples).

    Returns:
        A pandas DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    expression = None
    if filters:
        for col, op, val in filters:
            if op not in _OP_MAP:
                raise ValueError(f"Unsupported operator: {op}")
            clause = _OP_MAP[op](ds.field(col), val)
            expression = clause if expression is None else expression & clause

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(filter=expression)
    return table.to_pandas()


def save_parquet_dataset(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    compression: str = "snappy",
    max_rows_per_file: Optional[int] = 10000
) -> None:
    """
    Save a DataFrame as a partitioned Parquet dataset (folder of .parquet files).

    Args:
        df: Pandas DataFrame to save.
        output_dir: Target directory to save the dataset.
        compression: Compression type ('snappy', 'gzip', 'brotli', etc.).
        max_rows_per_file: Splits into multiple files of this row count (default to 10,000).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df)

    if max_rows_per_file:
        for i in range(0, table.num_rows, max_rows_per_file):
            chunk = table.slice(i, max_rows_per_file)
            pq.write_table(
                chunk,
                output_dir / f"part_{i//max_rows_per_file:03d}.parquet",
                compression=compression
            )
    else:
        pq.write_table(
            table,
            output_dir / "part_000.parquet",
            compression=compression
        )


def running_inside_docker():
    """Detect if the process is running inside a Docker container."""
    path = '/proc/self/cgroup'
    if os.path.exists('/.dockerenv'):
        return True
    if os.path.isfile(path):
        with open(path) as f:
            return any('docker' in line for line in f)
    return False
