import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModel
import config
from pathlib import Path
import json
import pandas as pd

from src.utils import load_params
from src.utils import get_tqdm
from src.logger import get_logger

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

logger = get_logger(__name__, level=logging.INFO)

params = load_params()
embed_config = params["embedding"]


def chunk_text(
        text: str,
        tokenizer,
        chunk_size,
        max_chunks: int,
        prefix_len: int = 0
) -> dict:

    sentences = sent_tokenize(text=text)
    sentences_norm = []
    for sent in sentences:
        tokenized = tokenizer.encode(
            sent.strip(), add_special_tokens=False)
        if len(tokenized) > chunk_size - prefix_len:
            split_size = chunk_size - prefix_len - 5  # -5 to be on the safe side
            subsents = [tokenizer.decode(tokenized[k:k+split_size])
                        for k in range(0, len(tokenized), split_size)]
            sentences_norm.extend(subsents)
        else:
            sentences_norm.append(sent)

    for sent in sentences_norm:
        tokenized = tokenizer.encode(
            sent.strip(), add_special_tokens=False)
        if len(tokenized) > chunk_size - prefix_len:
            print(sent)
            raise Exception(
                "There's still a text chunk larger than chunk_size")

    chunks = []
    chunks_len = []
    current_chunk = []
    current_len = 0
    for sent in sentences_norm:
        sent_clean = sent.strip()
        n_word_sent = len(sent_clean.split(' '))
        if n_word_sent > 2:
            tokenized = tokenizer.encode(
                sent_clean, add_special_tokens=False)
            if current_len + len(tokenized) < chunk_size - prefix_len:
                current_chunk.append(sent_clean)
                current_len += len(tokenized)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    chunks_len.append(current_len)
                current_chunk = [sent_clean]
                current_len = len(tokenized)

    if (current_chunk and current_len > chunk_size // 3
            and current_len < chunk_size - prefix_len):
        chunks.append(' '.join(current_chunk))
        chunks_len.append(current_len)

    overlapping_chunks = []
    window_masks = []
    for i in range(len(chunks)):
        if len(overlapping_chunks) > max_chunks:
            break

        mask = torch.zeros(tokenizer.model_max_length,
                           dtype=torch.float32)
        if i > 0 and i < len(chunks)-1:
            overlapping_chunks.append(' '.join(chunks[i-1:i+2]))
            start_idx = chunks_len[i-1]+prefix_len
            end_idx = start_idx + chunks_len[i]
            mask[start_idx:end_idx] = 1.0
        elif i == 0:
            overlapping_chunks.append(' '.join(chunks[i:i+3]))
            start_idx = prefix_len
            end_idx = start_idx + chunks_len[i]
            mask[start_idx:end_idx] = 1.0
        else:
            overlapping_chunks.append(' '.join(chunks[i-2:i+1]))
            start_idx = chunks_len[i-2]+chunks_len[i-1]+prefix_len
            end_idx = start_idx + chunks_len[i]
            mask[start_idx:end_idx] = 1.0
        if mask.sum() == 0:
            print(i, len(chunks), start_idx, end_idx)
            print(chunks[i-1:i+2])
            raise Exception("Mask's empty!")
        window_masks.append(mask.unsqueeze(0))

    return {"chunks": overlapping_chunks,
            "masks": window_masks,
            "core_chunks": chunks}


def load_and_chunk(paperData: dict,
                   tokenizer: AutoTokenizer,
                   chunk_size: int,
                   max_chunks: int,
                   prefix_len: int = 0):
    masked_chunks = chunk_text(paperData["main_text"], tokenizer,
                               chunk_size=chunk_size,
                               max_chunks=max_chunks,
                               prefix_len=prefix_len)
    texts = masked_chunks["chunks"]
    masks = masked_chunks["masks"]
    paper_id = paperData["paperId"]
    paperInfo = paperData.copy()
    payloads = [{**paperInfo, "text": chunk}
                for chunk in masked_chunks["chunks"]]

    return paper_id, texts, masks, payloads


def save_embeddings(paper_id: str,
                    data: dict,
                    cache_dir: Path):
    file_path = cache_dir / f"{paper_id}.json"
    with open(file_path.with_suffix(".inprogress"), "w") as f:
        json.dump(data, f)
    file_path.with_suffix(".inprogress").rename(file_path)


class PaperEmbed:
    def __init__(self,
                 model_name: str = embed_config['model_name'],
                 chunk_size: int = embed_config['chunk_size'],
                 max_chunks: int = embed_config['max_chunks'],
                 chunk_batch_size: int = embed_config['chunk_batch_size'],
                 paper_batch_size: int = embed_config['paper_batch_size'],
                 n_jobs: int = embed_config['n_jobs']):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, self.tokenizer = self.load_model()

        self.chunk_size = min(chunk_size, self.tokenizer.model_max_length//3
                              ) if chunk_size else self.tokenizer.model_max_length//3
        self.max_chunks = max_chunks
        self.chunk_batch_size = chunk_batch_size
        self.paper_batch_size = paper_batch_size
        self.cache_dir = config.VECTORDB_PATH / "tmp"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(
            self.model_name).to(self.device).eval()
        return model, tokenizer

    def unload_model(self):
        del self.model
        torch.cuda.empty_cache()

    def _model_prefix(self,
                      tokenized: bool = False,
                      role: str = 'document'):
        prefix = " "
        if "e5" in self.model_name:
            if role.lower() == 'document':
                prefix = "passage: "
            elif role.lower() == 'query':
                prefix = "query: "
            if tokenized:
                prefix = self.tokenizer.encode(prefix)[:-1]
        return prefix

    def compute_embeddings(self,
                           texts,
                           masks=None,
                           prefix: str = '',
                           suffix: str = ''):
        embeddings = []
        if masks is not None:
            masks = torch.cat(masks, dim=0)
        texts = [prefix + txt + suffix for txt in texts]
        tqdm = get_tqdm()
        with tqdm(range(0, len(texts), self.chunk_batch_size),
                  desc="Computing embeddings", colour='#b3ffb3') as pbar:
            for i in pbar:
                batch = texts[i:i+self.chunk_batch_size]
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True).to(self.device)

                if masks is not None:
                    mask = masks[i:i +
                                 self.chunk_batch_size].to(self.device).unsqueeze(-1)
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    S = outputs.last_hidden_state.size()[1]
                    masked_embeds = outputs.last_hidden_state * mask[:, :S, :]
                    embeds = masked_embeds.sum(1) / mask.sum(1)
                    embeddings.append(embeds.cpu())
                del embeds, masked_embeds, mask, outputs, inputs
                torch.cuda.empty_cache()

            if embeddings:
                embeddings = torch.cat(embeddings, dim=0)
            else:
                embeddings = None

        return embeddings

    def batch_embeddings(self, papers):
        save_executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        chunk_executor = ThreadPoolExecutor(max_workers=self.n_jobs)

        if isinstance(papers, pd.DataFrame):
            papers = papers.to_dict(orient='records')

        tqdm = get_tqdm()
        with tqdm(range(0, len(papers), self.paper_batch_size),
                  desc="Paper batches", position=0, colour='#000000') as pbar:
            for i in pbar:
                paperBatch = papers[i:i+self.paper_batch_size]
                doc_prefix = self._model_prefix(tokenized=True)
                futures = [chunk_executor.submit(
                    load_and_chunk, paper, self.tokenizer,
                    self.chunk_size, self.max_chunks, len(doc_prefix)) for paper in paperBatch]

                batched_texts = []
                batched_ids = []
                batched_masks = []
                batched_payloads = []
                all_paper_ids = []
                chunk_counts = []
                for future in as_completed(futures):
                    try:
                        paper_id, texts, masks, payloads = future.result()
                        if len(texts) > 1:
                            batched_ids.extend([paper_id]*len(texts))
                            batched_texts.extend(texts)
                            batched_masks.extend(masks)
                            batched_payloads.extend(payloads)
                            all_paper_ids.append(paper_id)
                            chunk_counts.append(len(texts))
                    except Exception as e:
                        print(f"Failed to chunk")
                        raise e

                sorted_batch = sorted(zip(batched_texts, batched_masks, batched_payloads,
                                          batched_ids), key=lambda x: len(x[0]))
                if sorted_batch:
                    batched_texts, batched_masks, batched_payloads, batched_ids = list(
                        zip(*sorted_batch))

                    doc_prefix = self._model_prefix(tokenized=False)
                    embeddings = self.compute_embeddings(texts=batched_texts,
                                                         masks=batched_masks,
                                                         prefix=doc_prefix)

                for paper_id in all_paper_ids:
                    embeded_data = {
                        "embeddings": [
                            emb.tolist() for id, emb in zip(batched_ids, embeddings)
                            if id == paper_id],
                        "payloads": [
                            payload for id, payload in zip(batched_ids, batched_payloads)
                            if id == paper_id]}
                    if embeded_data["embeddings"]:
                        save_executor.submit(
                            save_embeddings, paper_id, embeded_data, self.cache_dir)

        chunk_executor.shutdown(wait=True)
        save_executor.shutdown(wait=True)

    def embed_queries(self, queries):
        query_prefix = self._model_prefix(tokenized=False,
                                          role='query')
        embeddings = self.compute_embeddings(texts=queries,
                                             prefix=query_prefix)

        return embeddings
