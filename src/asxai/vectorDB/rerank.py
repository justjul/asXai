import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoConfig
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pytorch

import os
import json
import config

from asxai.dataIO import load_data
from asxai.vectorDB import QdrantManager

from typing import Union, List
from asxai.utils import load_params
from asxai.utils import log_and_register_model, auto_promote_best_model, clean_runs_keep_top_k, set_mlflow_uri
from asxai.utils import AsyncRunner
from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
reranking_config = params["reranking"]
qdrant_config = params["qdrant"]

async_runner = AsyncRunner()


class RerankEncoder(nn.Module):
    def __init__(self,
                 name: str = "reranker-default",
                 nhead: int = reranking_config['nhead'],
                 num_layers: int = reranking_config['num_layers'],
                 dropout: float = reranking_config['dropout'],
                 max_len: int = reranking_config['max_len'],
                 temperature: float = reranking_config['temperature'],
                 lr: float = reranking_config["learning_rate"],
                 qdrant_model: str = qdrant_config["model_name"],):
        super().__init__()
        # learnable positional embeddings

        self._config = {"name": name,
                        "nhead": nhead,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "max_len": max_len,
                        "temperature": temperature,
                        "lr": lr,
                        "qdrant_model": qdrant_model, }
        for key, value in self._config.items():
            setattr(self, key, value)

        self.emb_dim = AutoConfig.from_pretrained(
            self.qdrant_model).hidden_size

        self.pos_emb = nn.Parameter(torch.zeros(max_len, self.emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=self.emb_dim,
                                                   dropout=dropout,
                                                   batch_first=True)

        self._init_identity_transformer_layer(encoder_layer)

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        # optional scoring projection
        self.projection = nn.Linear(self.emb_dim, self.emb_dim)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.device = device

    def _init_identity_transformer_layer(self, layer):
        # Feedforward weights
        nn.init.zeros_(layer.linear1.weight)
        nn.init.zeros_(layer.linear1.bias)
        nn.init.zeros_(layer.linear2.weight)
        nn.init.zeros_(layer.linear2.bias)

        # Self-attention weights
        for name, param in layer.self_attn.named_parameters():
            if "weight" in name:
                nn.init.zeros_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # LayerNorm weights = 1, bias = 0
        for ln in [layer.norm1, layer.norm2]:
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        """
        x: (batch_size, num_chunks, emb_dim)
        returns: (batch_size, num_chunks, emb_dim)
        """
        if not isinstance(x, torch.Tensor):
            x = pad(x).to(self.device)
        seq_len = x.size(1)
        x = x + self.pos_emb[:seq_len]
        x = self.transformer(x)
        return self.projection(x)

    def compute_triplet_loss(self,
                             Q: torch.Tensor,
                             P: torch.Tensor,
                             N: torch.Tensor,
                             return_scores: bool = False):
        B, _, D = Q.shape
        Q = F.normalize(Q, p=2, dim=-1)
        P = F.normalize(P, p=2, dim=-1)
        N = F.normalize(N, p=2, dim=-1)

        def pseudo_max_cos_sim(A, B):  # A: (B, Ta, D), B: (B, Tb, D)
            sim = torch.bmm(A, B.transpose(1, 2))
            sim_max = torch.logsumexp(
                sim / self.temperature, dim=2)  # shape: (B, Ta)
            return sim_max.mean(dim=1)  # average over Ta → (B,)

        pos_score = pseudo_max_cos_sim(Q, P)  # (B,)
        neg_score = pseudo_max_cos_sim(Q, N)  # (B,)

        logits = torch.stack([pos_score, neg_score], dim=1)  # (B, 2)
        labels = torch.zeros(B, dtype=torch.long, device=Q.device)
        loss = F.cross_entropy(logits, labels)
        if return_scores:
            pos_score = self.compute_max_sim(Q, P)
            neg_score = self.compute_max_sim(Q, N)
            return loss, pos_score, neg_score
        return loss

    def compute_max_sim(self,
                        Q_embeds: torch.Tensor,
                        D_embeds: torch.Tensor):
        if D_embeds.size(1) == 0:
            return torch.zeros(Q_embeds.size(0), device=Q_embeds.device)
        logger.info(f"{Q_embeds.size()}, {D_embeds.size()}")
        if Q_embeds.dim() == 2:
            Q_embeds = Q_embeds.unsqueeze(0)
        if D_embeds.dim() == 2:
            D_embeds = D_embeds.unsqueeze(0)

        Q_embeds = F.normalize(Q_embeds, p=2, dim=-1)
        D_embeds = F.normalize(D_embeds, p=2, dim=-1)
        if Q_embeds.size(0) == 1 and D_embeds.size(0) > 1:
            Q_embeds = Q_embeds.expand(D_embeds.size(0), -1, -1)
        max_sim = (torch.bmm(Q_embeds, D_embeds.transpose(1, 2))).max(
            dim=-1).values.sum(dim=1)
        return max_sim

    def rerank_score(self,
                     Q_embeds: Union[List, torch.Tensor],
                     D_embeds: Union[List, torch.Tensor],
                     skip_rerank: bool = False):
        if not isinstance(Q_embeds, torch.Tensor):
            Q_embeds = torch.tensor(Q_embeds)
        if not isinstance(D_embeds, torch.Tensor):
            D_embeds = torch.tensor(D_embeds)

        if Q_embeds.dim() == 2:
            Q_embeds = Q_embeds.unsqueeze(0)
        if D_embeds.dim() == 2:
            D_embeds = D_embeds.unsqueeze(0)

        if not skip_rerank:
            device = next(self.parameters()).device
            with torch.no_grad():
                Q_embeds = self.forward(Q_embeds.to(device=device))
                D_embeds = self.forward(D_embeds.to(device=device))
        else:
            Q_embeds = Q_embeds.to(dtype=torch.float32)
            D_embeds = D_embeds.to(dtype=torch.float32)

        max_sim = self.compute_max_sim(Q_embeds, D_embeds)

        return max_sim.squeeze(0).cpu().tolist()

    def train_reranker_from_cite(self,
                                 years_range: List[int] = reranking_config["years_cite_range"]):
        if not isinstance(years_range, list):
            years_range = [years_range]
        years = [years for years in range(
            min(years_range), max(years_range)+1)]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        rerankDataset = RerankerDataset(years=years)

        logger.setLevel(config.LOG_LEVEL)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        train_reranker(self, rerankDataset, optimizer, device=device)

    def save(self):
        model_dir = config.MODELS_PATH / self.name
        os.makedirs(model_dir, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(model_dir, "model.pt"))

        model_config = {k: getattr(self, k) for k in self._config}
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(model_config, f)

    @classmethod
    def load(cls, model_name: str = "reranker-default", version: int = None):
        set_mlflow_uri()
        client = MlflowClient()

        logger.setLevel(config.LOG_LEVEL)
        if version is None:
            # Load the version tagged with 'status' = 'production'
            logger.info(f"Will load {model_name} with status production")
            versions = client.search_model_versions(f"name='{model_name}'")
            prod_versions = [
                v for v in versions
                if v.tags.get("status", "").lower() == "production"
            ]
            if not prod_versions:
                logger.error(f"No production model found for '{model_name}'")
                return None
            best_version = max(prod_versions, key=lambda v: int(v.version))
            version = best_version.version

        logger.info(f"Loading {model_name} - version-{version}")

        model_uri = f"models:/{model_name}/{version}"

        try:
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"{model_uri} loaded")
            model.version = str(version)

            model_version_info = client.get_model_version(
                name=model_name, version=version)
            training_date = model_version_info.tags.get("training_date")
            if training_date:
                model.training_date = training_date
            else:
                model.training_date = "unknown"
        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {e}")
            return None

        if not isinstance(model, cls):
            logger.error(
                f"Loaded model is of type {type(model)}, expected {cls.__name__}")
            return None

        return model


async def triplets_from_cite(qdrant,
                             years: Union[int, List[int]],
                             deltaMonths: int = reranking_config['cite_deltaMonths'],
                             topK_near_cite_range: List[int] = reranking_config['topK_near_cite_range']):
    paperdata = load_data(years,
                          data_types=['text', 'metadata'],
                          filters=[('pdf_status', '==', 'extracted')])
    all_paperIds = paperdata['metadata']['paperId'].to_list()
    cited_refs = [ref for ref_list in paperdata['metadata']
                  ['referenceIds'] for ref in ref_list.split(';') if ref != '']
    cited_refs = set(cited_refs).intersection(all_paperIds)

    positive_pairs = [(paperdata['metadata']['publicationDate'].iloc[i], paperdata['metadata']['paperId'].iloc[i], [
                       ref for ref in ref_list.split(';') if ref in cited_refs]) for i, ref_list in enumerate(paperdata['metadata']['referenceIds'])]

    positive_pairs = [p for p in positive_pairs if p[2]]

    res = await qdrant.client.get_collection(qdrant.collection_name_ids)
    vector_size = res.config.params.vectors.size

    triplets = []
    for pair in positive_pairs:
        pubdate, paper_Id, ref_Ids = pair
        date_obj = datetime.strptime(pubdate, "%Y-%m-%d")
        three_months_before = date_obj - relativedelta(months=deltaMonths)
        date_lim = three_months_before.strftime("%Y-%m-%d")
        for ref_id in ref_Ids:
            res = await qdrant.query_batch_streamed(query_vectors=[np.random.randn(vector_size).tolist()],
                                                    topKs=50,
                                                    topK_per_paper=0,
                                                    payload_filters=[[
                                                        ['paperId', '==', ref_id]]],
                                                    with_vectors=True,
                                                    )
            if res[0].points:
                ref_embed = res[0].points[0].vector
                res = await qdrant.query_batch_streamed(query_vectors=[ref_embed],
                                                        topKs=1,
                                                        topK_per_paper=0,
                                                        offset=np.random.randint(
                                                            *topK_near_cite_range),
                                                        payload_filters=[[['publicationDate', 'lt', date_lim],
                                                                         ['paperId', '!=', ref_Ids + [paper_Id]]]],
                                                        )
                if res[0].points:
                    pos_id = ref_id
                    neg_id = res[0].points[0].payload['paperId']
                    triplets.append((paper_Id, pos_id, neg_id))
    return triplets


class RerankerDataset(Dataset):
    def __init__(self,
                 query_embed=None,
                 query_Ids=None,
                 positive_Ids=None,
                 negative_Ids=None,
                 years=None,
                 qdrant_model: str = qdrant_config["model_name"],):
        self.query_embed = query_embed
        self.query_Ids = query_Ids if query_embed else None
        self.positive_Ids = positive_Ids
        self.negative_Ids = negative_Ids
        self.years = years or [2024, 2025]
        self.qdrant = QdrantManager()

        self.vector_size = AutoConfig.from_pretrained(qdrant_model).hidden_size

        self.async_runner = async_runner

        if not (self.query_embed or self.query_Ids):
            self.triplets = self.async_runner.run(
                triplets_from_cite(self.qdrant, self.years))

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
        return (torch.tensor(embeds[q_id], dtype=torch.float32),
                torch.tensor(embeds[p_id], dtype=torch.float32),
                torch.tensor(embeds[n_id], dtype=torch.float32))


def pad(tensors):
    max_len = max([x.size(0) for x in tensors])
    emb_dim = tensors[0].size(1)
    out = torch.zeros(len(tensors), max_len, emb_dim)
    for i, x in enumerate(tensors):
        out[i, :x.size(0)] = x
    return out


def collate_fn(batch):
    """
    Pads sequences in batch to the longest chunk length.
    Returns: (Q, P, N) padded to shape (B, T_max, D)
    """
    q_list, p_list, n_list = zip(*batch)
    return pad(q_list), pad(p_list), pad(n_list)


def train_reranker(model,
                   dataset,
                   optimizer,
                   device,
                   epochs: int = reranking_config['training_epochs'],
                   test_size: float = reranking_config['test_size']):
    set_mlflow_uri()
    mlflow.set_experiment(f"{model.name}")
    with mlflow.start_run(run_name=f"{model.name}"):
        mlflow.log_params({
            "nhead": model.nhead,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "max_len": model.max_len,
            "temperature": model.temperature,
            "learning_rate": model.lr
        })

        generator = torch.Generator().manual_seed(123)
        test_size = max(1, int(test_size * len(dataset)))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=generator)
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=16,
                                 shuffle=False, collate_fn=collate_fn)

        model.to(device)
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for Q, P, N in train_loader:
                Q, P, N = Q.to(device), P.to(device), N.to(device)
                Q_out = model(Q)
                P_out = model(P)
                N_out = model(N)
                loss = model.compute_triplet_loss(Q_out, P_out, N_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / len(train_loader)

            model.eval()
            test_loss = 0.0
            total_correct = 0
            total_margin = 0.0
            total_samples = 0
            with torch.no_grad():
                for Q, P, N in test_loader:
                    Q, P, N = Q.to(device), P.to(device), N.to(device)
                    Q_out = model(Q)
                    P_out = model(P)
                    N_out = model(N)
                    loss, pos_scores, neg_scores = model.compute_triplet_loss(
                        Q_out, P_out, N_out, return_scores=True)
                    test_loss += loss.item()
                    total_correct += (pos_scores > neg_scores).sum().item()
                    total_margin += (pos_scores - neg_scores).sum().item()
                    total_samples += Q.size(0)
                avg_test = test_loss / len(train_loader)
                acc = total_correct / total_samples
                margin = total_margin / total_samples
            logger.info(
                f"Epoch {epoch+1} - Training Avg Loss: {avg_train:.4f} - Test Avg Loss: {avg_test:.4f} - Acc: {acc:.3f} - Margin: {margin:.4f}")

            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("test_loss", avg_test, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("margin", margin, step=epoch)

        log_and_register_model(model=model,
                               model_name=model.name,
                               input_tensor=Q,
                               output_tensor=Q_out,
                               status="staging",
                               metrics={
                                   "accuracy": acc,
                                   "margin": margin,
                                   "test_loss": avg_test},)

        auto_promote_best_model(name=model.name, metric="margin")

        clean_runs_keep_top_k(model_name=model.name, k=3, metric="margin")
