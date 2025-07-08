import sys
import yaml
from pathlib import Path
from tqdm import tqdm as std_tqdm
from tqdm.notebook import tqdm as nb_tqdm
import os
import shutil
from datetime import datetime

import pyarrow.dataset as ds
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union, Optional, List, Tuple
import operator

import config

import mlflow.pytorch
import asyncio
import threading
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

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


def set_mlflow_uri():
    hostname = "mlflow" if running_inside_docker() else "localhost"
    mlflow.set_tracking_uri(f"http://{hostname}:5000")
    # mlflow.set_registry_uri()


def log_and_register_model(
    model,
    model_name: str,
    input_tensor,
    output_tensor,
    status: str = "staging",  # e.g., "production", "staging"
    metrics: dict = None,
):
    # Convert and infer signature
    input_example = input_tensor.detach().cpu().numpy().astype("float32")
    output_example = output_tensor.detach().cpu().numpy().astype("float32")
    signature = infer_signature(input_example, output_example)

    artifact_path = getattr(model, "name", model_name)
    set_mlflow_uri()
    mlflow.pytorch.log_model(
        model,
        artifact_path=artifact_path,
        input_example=input_example,
        signature=signature,
    )

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered_name = f"{model_name}"

    # Register model
    result = mlflow.register_model(model_uri=model_uri, name=registered_name)

    client = MlflowClient()

    # Set tags
    version = result.version
    if metrics:
        desc = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        client.update_model_version(name=registered_name,
                                    version=version,
                                    description=f"Metrics: {desc}")
        for k, v in metrics.items():
            client.set_model_version_tag(registered_name, version, k, str(v))

    # Replace "stage" with tag
    client.set_model_version_tag(registered_name,
                                 version,
                                 "status", status.lower())
    date_str = datetime.now().strftime("%Y-%m-%d")
    client.set_model_version_tag(
        registered_name, version, "training_date", date_str)

    print(
        f"Registered '{registered_name}' as version {version} with status='{status}'.")


def auto_promote_best_model(name: str, metric: str = "accuracy"):
    set_mlflow_uri()
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")

    # Include staging + current production versions
    candidates = [
        v for v in versions
        if v.tags.get("status", "").lower() in {"staging", "production"}
        and metric in v.tags
    ]

    if not candidates:
        print("No versions with status and metric found.")
        return

    # Select version with highest metric
    best_version = max(candidates, key=lambda v: float(v.tags[metric]))
    best_vnum = best_version.version
    best_score = float(best_version.tags[metric])

    # Remove production tag from others
    for v in candidates:
        if v.version != best_vnum and v.tags.get("status", "").lower() == "production":
            client.delete_model_version_tag(name, v.version, "status")
            print(f"Removed production tag from version {v.version}")

    # Tag the best version as production
    client.set_model_version_tag(name, best_vnum, "status", "production")
    print(
        f"Promoted version {best_vnum} to production ({metric} = {best_score:.4f})")


def clean_runs_keep_top_k(model_name, k=3, metric="accuracy"):
    set_mlflow_uri()
    client = MlflowClient()
    experiments = client.search_experiments()
    experiment_ids = [
        e.experiment_id for e in experiments if e.name == model_name]
    all_runs = []

    for exp_id in experiment_ids:
        runs = client.search_runs([exp_id], max_results=1000)
        for run in runs:
            if run.data.tags.get('mlflow.runName') == model_name:
                all_runs.append(run)

    if not all_runs:
        print(f"No runs found for model '{model_name}'")
        return

    valid_runs = [r for r in all_runs if metric in r.data.metrics]
    best_runs = sorted(valid_runs, key=lambda r: -r.data.metrics[metric])[:k]
    most_recent = max(all_runs, key=lambda r: r.info.start_time)

    # Fetch all registered model versions
    registered_versions = client.search_model_versions(f"name='{model_name}'")

    # Always keep registered versions linked to top K or most recent
    keep_run_ids = {
        *[r.info.run_id for r in best_runs],
        most_recent.info.run_id,
    }

    # Keep model versions whose run_id matches
    keep_versions = [
        v for v in registered_versions if v.run_id in keep_run_ids
    ]
    delete_versions = [
        v for v in registered_versions if v.run_id not in keep_run_ids
    ]

    # Delete unnecessary registered versions
    try:
        all_model_path = config.MODELS_PATH / "mlflow_storage" / "mlruns" / "models"
        for v in delete_versions:
            delete_model_path = all_model_path / \
                model_name / f"version-{v.version}"
            if os.path.isdir(delete_model_path):
                shutil.rmtree(delete_model_path)
            print(f"Deleted model version {v.version} (run_id={v.run_id})")
            # client.delete_model_version(name=model_name, version=v.version)

        # Delete runs and artifacts not in keep_run_ids
        all_exp_path = config.MODELS_PATH / "mlflow_storage" / "mlruns"
        all_artifact_path = config.MODELS_PATH / "mlflow_storage" / "artifacts"
        for run in all_runs:
            if run.info.run_id not in keep_run_ids:
                delete_run_path = all_exp_path / \
                    str(run.info.experiment_id) / str(run.info.run_id)
                if os.path.isdir(delete_run_path):
                    shutil.rmtree(delete_run_path)
                delete_artifact_path = all_artifact_path / \
                    str(run.info.experiment_id) / str(run.info.run_id)
                if os.path.isdir(delete_artifact_path):
                    shutil.rmtree(delete_artifact_path)
                print(f"Deleted run {run.info.run_id}")
                # client.delete_run(run.info.run_id)

        # Clean up trash folder
        mlruns_trash_path = config.MODELS_PATH / "mlflow_storage" / "mlruns" / ".trash"
        for d in os.listdir(mlruns_trash_path):
            deleted_path = os.path.join(mlruns_trash_path, d)
            if os.path.isdir(deleted_path):
                shutil.rmtree(deleted_path)
    except Exception as e:
        print(e)
        print(
            f"YOU SHOULD RUN: sudo chmod -R 777 on {all_model_path.parent.parent}")
