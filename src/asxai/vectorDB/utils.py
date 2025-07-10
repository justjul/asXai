import sys
import os
import shutil
from datetime import datetime

from typing import Union, Optional, List, Tuple

import config

import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from asxai.utils import running_inside_docker

import torch
from torch.utils.data import Dataset
import numpy as np
from dateutil.relativedelta import relativedelta


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
        name=artifact_path,
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
        # Adding description doesn't seem to work anymore, without any clear reason
        # desc = " ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        # client.update_model_version(name=registered_name,
        #                             version=version,
        #                             description=f"Metrics: {desc}")
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


async def triplets_from_cite(
    paperdata: dict,
    qdrant,
    deltaMonths: int = None,
    topK_near_cite_range: List[int] = None,
    model_type: str = None
):
    all_paperIds = paperdata['metadata']['paperId'].to_list()

    # WARNING: There may be an issue here if arXiv paper Ids replace those from s2.
    # But for now we'll just ignore it
    cited_refs = [ref for ref_list in paperdata['metadata']
                  ['referenceIds'] for ref in ref_list.split(';') if ref != '']
    cited_refs = set(cited_refs).intersection(all_paperIds)

    positive_pairs = [(paperdata['metadata']['publicationDate'].iloc[i], paperdata['metadata']['paperId'].iloc[i], [
                       ref for ref in ref_list.split(';') if ref in cited_refs]) for i, ref_list in enumerate(paperdata['metadata']['referenceIds'])]

    positive_pairs = [p for p in positive_pairs if p[2]]

    res = await qdrant.client.get_collection(qdrant.collection_name_ids)
    vector_size = res.config.params.vectors.size

    triplets = []
    for pubdate, paper_Id, ref_Ids in positive_pairs:
        if pubdate and pubdate != 'None':
            date_obj = datetime.strptime(pubdate, "%Y-%m-%d")
            delta_months_before = date_obj - relativedelta(months=deltaMonths)
            date_lim = delta_months_before.strftime("%Y-%m-%d")

            res = await qdrant.query_batch_streamed(query_vectors=[np.random.randn(vector_size).tolist()],
                                                    topKs=1,
                                                    topK_per_paper=0,
                                                    payload_filters=[[
                                                        ['paperId', '==', paper_Id]]],
                                                    with_vectors=False,
                                                    )
            if not res[0].points:
                continue

            for ref_id in ref_Ids:
                res = await qdrant.query_batch_streamed(query_vectors=[np.random.randn(vector_size).tolist()],
                                                        topKs=50,
                                                        topK_per_paper=0,
                                                        payload_filters=[[
                                                            ['paperId', '==', ref_id]]],
                                                        with_vectors=True,
                                                        )
                if not res[0].points:
                    continue

                ref_embed = res[0].points[0].vector
                res = await qdrant.query_batch_streamed(query_vectors=[ref_embed],
                                                        topKs=1,
                                                        topK_per_paper=0,
                                                        offset=np.random.randint(
                                                            *topK_near_cite_range),
                                                        payload_filters=[[['publicationDate', 'lt', date_lim],
                                                                          ['paperId', '!=', ref_Ids + [paper_Id]]]],
                                                        )
                if not res[0].points:
                    continue

                pos_id = ref_id
                neg_id = res[0].points[0].payload['paperId']
                if model_type.lower() == 'reranker':
                    triplets.append((paper_Id, pos_id, neg_id))
                elif model_type.lower() == 'innovator':
                    triplets.append((pos_id, paper_Id, neg_id))
    return triplets


class CiteDataset(Dataset):
    def __init__(
        self,
        qdrantManager,
        query_embed=None,
        query_Ids=None,
        positive_Ids=None,
        negative_Ids=None,
        model_type: str = None,
    ):
        self.query_embed = query_embed
        self.query_Ids = query_Ids if query_embed else None
        self.positive_Ids = positive_Ids
        self.negative_Ids = negative_Ids
        self.model_type = model_type
        self.qdrant = qdrantManager
        self.vector_size = qdrantManager.vector_size

        self.async_runner = AsyncRunner()

    def buildTriplets(
        self,
        paperdata: dict,
        deltaMonths: int = None,
        topK_near_cite_range: List[int] = None,
    ):
        self.triplets = self.async_runner.run(
            triplets_from_cite(
                paperdata,
                self.qdrant,
                deltaMonths=deltaMonths,
                topK_near_cite_range=topK_near_cite_range,
                model_type=self.model_type))

    def _get_qdrant_embeddings(self, paperIds):
        async def run_qdrant(paperIds):
            res = await self.qdrant.query_batch_streamed(
                query_vectors=[np.random.randn(self.vector_size).tolist()],
                topKs=3,
                topK_per_paper=0,
                payload_filters=[[['paperId', '==', paperIds]]],
                with_vectors=True,
            )
            embeds = {pt.payload["paperId"]: pt.vector for pt in res[0].points}
            return embeds

        return self.async_runner.run(run_qdrant(paperIds))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        q_id, p_id, n_id = self.triplets[idx]
        embeds = self._get_qdrant_embeddings([q_id, p_id, n_id])
        for k in embeds:
            embeds[k] = torch.tensor(embeds[k], dtype=torch.float32)
        if self.model_type == 'innovator':
            embeds[p_id] = embeds[p_id][0, :]
            embeds[n_id] = embeds[n_id][0, :]
        return (embeds[q_id],
                embeds[p_id],
                embeds[n_id])
