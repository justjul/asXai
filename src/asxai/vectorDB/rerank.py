import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoConfig
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pytorch

import os
import json
import config

from typing import Union, List
from asxai.dataIO import load_data
from asxai.vectorDB import QdrantManager
from asxai.utils import load_params
from asxai.vectorDB.utils import log_and_register_model, auto_promote_best_model, clean_runs_keep_top_k, set_mlflow_uri
from asxai.vectorDB.utils import CiteDataset
from asxai.utils import AsyncRunner
from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
reranking_config = params["reranking"]
innovating_config = params["innovating"]
qdrant_config = params["qdrant"]

async_runner = AsyncRunner()


def pad(tensors):
    if tensors[0].dim() < 2:
        return torch.stack(tensors)
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


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim))
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (B, T, D)
        scores = torch.einsum('btd,d->bt', self.linear(x),
                              self.query)  # (B, T)
        weights = F.softmax(scores, dim=1)  # (B, T)
        pooled = torch.einsum('btd,bt->bd', x, weights)  # (B, D)
        return pooled


def compute_max_sim(
    Q_embeds: torch.Tensor,
    D_embeds: torch.Tensor
):
    if D_embeds.size(1) == 0:
        return torch.zeros(Q_embeds.size(0), device=Q_embeds.device)
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


class BaseEncoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self._config = config_dict
        for k, v in config_dict.items():
            setattr(self, k, v)

        self.emb_dim = AutoConfig.from_pretrained(
            self.qdrant_model).hidden_size

        self.init_layers()

        self.to(self.device)

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def init_layers(self):
        self.pos_emb = nn.Parameter(torch.zeros(self.max_len, self.emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.nhead,
            dim_feedforward=self.emb_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        self._init_identity_transformer_layer(encoder_layer)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers)

        self.projection = nn.Linear(self.emb_dim, self.emb_dim)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def _init_identity_transformer_layer(self, layer):
        nn.init.zeros_(layer.linear1.weight)
        nn.init.zeros_(layer.linear1.bias)
        nn.init.zeros_(layer.linear2.weight)
        nn.init.zeros_(layer.linear2.bias)
        for name, param in layer.self_attn.named_parameters():
            if "weight" in name or "bias" in name:
                nn.init.zeros_(param)
        for ln in [layer.norm1, layer.norm2]:
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        if not isinstance(x, torch.Tensor):
            x = pad(x).to(self.device)
        seq_len = x.size(1)
        x = x + self.pos_emb[:seq_len]
        x = self.transformer(x)
        x = self.projection(x)
        return x

    def save(self):
        model_dir = config.MODELS_PATH / self.name
        os.makedirs(model_dir, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(model_dir, "model.pt"))

        model_config = {k: getattr(self, k) for k in self._config}
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(model_config, f)

    @classmethod
    def load(cls, model_name: str = None, version: int = None):
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


class RerankEncoder(BaseEncoder):
    def __init__(
        self,
        name: str = "reranker",
        nhead: int = reranking_config['nhead'],
        num_layers: int = reranking_config['num_layers'],
        dropout: float = reranking_config['dropout'],
        max_len: int = reranking_config['max_len'],
        temperature: float = reranking_config['temperature'],
        lr: float = reranking_config["learning_rate"],
        qdrant_model: str = qdrant_config["model_name"],
    ):
        config = {
            "name": name,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "max_len": max_len,
            "temperature": temperature,
            "lr": lr,
            "qdrant_model": qdrant_model,
        }
        super().__init__(config)

    def compute_triplet_loss(
        self,
        Q: torch.Tensor,
        P: torch.Tensor,
        N: torch.Tensor,
        return_scores: bool = False
    ):
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
            pos_score = compute_max_sim(Q, P)
            neg_score = compute_max_sim(Q, N)
            return loss, pos_score, neg_score
        return loss

    def rerank_score(
        self,
        Q_embeds: Union[List, torch.Tensor],
        D_embeds: Union[List, torch.Tensor],
        skip_rerank: bool = False
    ):
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

        max_sim = compute_max_sim(Q_embeds, D_embeds)

        return max_sim.squeeze(0).cpu().tolist()

    def train_from_cite(
        self,
        years_range: List[int] = reranking_config["years_cite_range"],
        from_scratch: str = reranking_config["train_from_scratch"]
    ):
        if not isinstance(years_range, list):
            years_range = [years_range]
        years = [years for years in range(
            min(years_range), max(years_range)+1)]

        if from_scratch:
            self.init_layers()

        paperdata = load_data(
            years, data_types=['text', 'metadata']
        )
        rerankDataset = CiteDataset(
            model_type='reranker',
            qdrantManager=QdrantManager()
        )
        rerankDataset.buildTriplets(
            paperdata,
            deltaMonths=reranking_config['cite_deltaMonths'],
            topK_near_cite_range=reranking_config['topK_near_cite_range']
        )

        logger.setLevel(config.LOG_LEVEL)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        train_encoder(
            self, dataset=rerankDataset, optimizer=optimizer, model_type='reranker')

    @classmethod
    def load(cls, model_name: str = "reranker", version: int = None):
        model = super().load(model_name=model_name, version=version)
        return model


class InnovEncoder(BaseEncoder):
    def __init__(
        self, name: str = "innovator",
        nhead: int = innovating_config['nhead'],
        num_layers: int = innovating_config['num_layers'],
        dropout: float = innovating_config['dropout'],
        max_len: int = innovating_config['max_len'],
        temperature: float = innovating_config['temperature'],
        lr: float = innovating_config["learning_rate"],
        qdrant_model: str = qdrant_config["model_name"],
    ):
        config = {
            "name": name,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "max_len": max_len,
            "temperature": temperature,
            "lr": lr,
            "qdrant_model": qdrant_model,
        }
        super().__init__(config)
        self.init_layers()

    def init_layers(self):
        super().init_layers()
        self.pool = AttentionPooling(input_dim=self.emb_dim)
        self.to(self.device)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        x = super().forward(x)  # (B, T, D)
        x = self.pool(x)        # (B, D)
        return x

    def compute_triplet_loss(
        self,
        Q: torch.Tensor,
        P: torch.Tensor,
        N: torch.Tensor,
        return_scores: bool = False,
    ):
        Q = F.normalize(Q, p=2, dim=-1)
        P = F.normalize(P, p=2, dim=-1)
        N = F.normalize(N, p=2, dim=-1)

        pos_sim = (Q * P).sum(dim=-1)  # (B,)
        neg_sim = (Q * N).sum(dim=-1)  # (B,)

        logits = torch.stack([pos_sim, neg_sim], dim=1)  # (B, 2)
        labels = torch.zeros(Q.size(0), dtype=torch.long, device=Q.device)
        loss = F.cross_entropy(logits, labels)
        if return_scores:
            pos_score = pos_sim
            neg_score = neg_sim
            return loss, pos_score, neg_score
        return loss

    def compute_cosine_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ):
        predicted = F.normalize(predicted, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        return 1 - (predicted * target).sum(dim=-1).mean()

    def compute_margin_cosine_loss(self, predicted, target, margin=0.2):
        predicted = F.normalize(predicted, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        sim = (predicted * target).sum(dim=-1)
        loss = (1 - sim).clamp(min=margin).mean()
        return loss

    def train_from_cite(
        self,
        years_range: List[int] = innovating_config["years_cite_range"],
        from_scratch: str = innovating_config["train_from_scratch"]
    ):
        if not isinstance(years_range, list):
            years_range = [years_range]
        years = [years for years in range(
            min(years_range), max(years_range)+1)]

        if from_scratch:
            self.init_layers()

        paperdata = load_data(
            years, data_types=['text', 'metadata']
        )
        innovDataset = CiteDataset(
            model_type='innovator',
            qdrantManager=QdrantManager()
        )
        innovDataset.buildTriplets(
            paperdata,
            deltaMonths=innovating_config['cite_deltaMonths'],
            topK_near_cite_range=innovating_config['topK_near_cite_range']
        )

        logger.setLevel(config.LOG_LEVEL)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        train_encoder(
            self, dataset=innovDataset, optimizer=optimizer, model_type='innovator')

    @classmethod
    def load(cls, model_name: str = "innovator", version: int = None):
        model = super().load(model_name=model_name, version=version)
        return model


def train_encoder(
    model,
    dataset,
    optimizer,
    model_type,
    epochs: int = reranking_config['training_epochs'],
    test_size: float = reranking_config['test_size']
):
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

        logger.info(
            f"will now train {model.name} on {train_size} samples for {epochs} epochs")

        device = model.device
        model.to(device)
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for Q, P, N in train_loader:
                Q, P, N = Q.to(device), P.to(device), N.to(device)
                Q_out = model(Q)
                if model_type == "reranker":
                    P_out = model(P)
                    N_out = model(N)
                elif model_type == "innovator":
                    P_out = P
                    N_out = N
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
                    if model_type == "reranker":
                        P_out = model(P)
                        N_out = model(N)
                    elif model_type == "innovator":
                        P_out = P
                        N_out = N
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
