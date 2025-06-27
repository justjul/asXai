from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer, AutoModel
import config
from pathlib import Path
import json
import pandas as pd
import re
import os

from asxai.utils import load_params
from asxai.utils import get_tqdm
from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
embed_config = params["embedding"]

sentence_splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def chunk_text(
        text: str,
        tokenizer,
        chunk_size,
        max_chunks: int,
        prefix_len: int = 0
) -> dict:

    sentences = [s.strip() for s in sentence_splitter.split(text) if s.strip()]

    tokenized = tokenizer(sentences, add_special_tokens=False)
    all_token_ids = tokenized["input_ids"]

    # Flatten and split long token sequences
    sentence_tokens = []
    max_len = chunk_size - prefix_len - 5
    for token_ids in all_token_ids:
        if len(token_ids) > max_len:
            sentence_tokens.extend(
                [token_ids[i:i+max_len]
                    for i in range(0, len(token_ids), max_len)]
            )
        else:
            sentence_tokens.append(token_ids)

    chunks_token_ids = []
    chunks_len = []
    current_chunk = []
    current_len = 0

    for token_ids in sentence_tokens:
        if len(token_ids) < 3:
            continue
        if current_len + len(token_ids) < chunk_size - prefix_len:
            current_chunk.extend(token_ids)
            current_len += len(token_ids)
        else:
            if current_chunk:
                chunks_token_ids.append(current_chunk)
                chunks_len.append(current_len)
            current_chunk = token_ids.copy()
            current_len = len(token_ids)
        if len(chunks_token_ids) > max_chunks + 2:
            break

    if len(chunks_token_ids) < max_chunks+2 and current_chunk and current_len > 0:
        chunks_token_ids.append(current_chunk)
        chunks_len.append(current_len)

    overlapping_token_chunks, core_token_chunks = [], []
    window_masks = []

    for i in range(len(chunks_token_ids)):
        if len(overlapping_token_chunks) > max_chunks:
            break

        mask = torch.zeros(tokenizer.model_max_length, dtype=torch.float32)

        prev_chunk = chunks_token_ids[i-1] if i > 0 else []
        curr_chunk = chunks_token_ids[i]
        next_chunk = chunks_token_ids[i+1] if i + \
            1 < len(chunks_token_ids) else []
        next_next_chunk = chunks_token_ids[i +
                                           2] if i+2 < len(chunks_token_ids) else []
        prev_prev_chunk = chunks_token_ids[i-2] if i-2 >= 0 else []

        # Choose context window
        if i > 0 and i < len(chunks_token_ids)-1:
            concat_chunks = prev_chunk + curr_chunk + next_chunk
            start_idx = len(prev_chunk) + prefix_len
        elif i == 0:
            concat_chunks = curr_chunk + next_chunk + next_next_chunk
            start_idx = prefix_len
        else:  # i is last index or second last
            concat_chunks = prev_prev_chunk + prev_chunk + curr_chunk
            start_idx = len(prev_prev_chunk) + len(prev_chunk) + prefix_len

        end_idx = start_idx + len(curr_chunk)
        if end_idx > tokenizer.model_max_length:
            end_idx = tokenizer.model_max_length  # Prevent overflow
        mask[start_idx:end_idx] = 1.0

        if mask.sum() == 0:
            print("Empty mask!", i, start_idx, end_idx)
            raise Exception("Empty mask!")

        overlapping_token_chunks.append(concat_chunks)
        window_masks.append(mask.unsqueeze(0))
        core_token_chunks.append(curr_chunk)

    overlapping_chunks = tokenizer.batch_decode(overlapping_token_chunks,
                                                skip_special_tokens=True)
    core_chunks = tokenizer.batch_decode(overlapping_token_chunks,
                                         skip_special_tokens=True)

    return {"chunks": overlapping_chunks,
            "core_chunks": core_chunks,
            "masks": window_masks}


def load_and_chunk(paperData: dict,
                   tokenizer: AutoTokenizer,
                   chunk_size: int,
                   max_chunks: int,
                   prefix_len: int = 0):
    masked_chunks = chunk_text(paperData["main_text"], tokenizer,
                               chunk_size=chunk_size,
                               max_chunks=max_chunks,
                               prefix_len=prefix_len)
    text_to_embed = masked_chunks["chunks"]
    masks = masked_chunks["masks"]
    paper_id = paperData["paperId"]
    paperInfo = paperData.copy()
    payloads = [{**paperInfo, "text": chunk, "is_ref": False}
                # masked_chunks["chunks"]]
                for chunk in masked_chunks["core_chunks"]]

    return paper_id, text_to_embed, masks, payloads


def get_normalized_textdata(textdata):
    papertext = textdata.copy()
    papertext['main_text'] = papertext['main_text'].fillna('')
    papertext['main_text'] = papertext['main_text'].replace(to_replace='None',
                                                            value='')
    papertext['pdf_extracted'] = papertext['main_text'].str.strip().str.len() > 500

    papertext['ref_text'] = papertext['ref_text'].fillna('')
    papertext['ref_text'] = papertext['ref_text'].replace(to_replace='None',
                                                          value='')

    mask = papertext["main_text"].str.len() < 500
    papertext.loc[mask, "main_text"] = papertext.loc[mask].apply(
        lambda x: ' '.join([x["title"], x["abstract"]]), axis=1)

    mask = papertext["ref_text"].str.len() < 500
    papertext.loc[mask, "ref_text"] = papertext.loc[mask].apply(
        lambda x: ' '.join([x["title"], x["abstract"]]), axis=1)

    papertext = papertext[['paperId', 'main_text', 'ref_text', 'pdf_extracted', 'title',
                           'abstract',]]
    return papertext


def get_normalized_metadata(metadata):
    return metadata


def get_payload_and_text(paperdata):
    textdata = get_normalized_textdata(paperdata["text"])
    metadata = get_normalized_metadata(paperdata["metadata"])
    data = pd.merge(metadata, textdata, how='left', on='paperId')

    return data


def save_embeddings(paper_id: str,
                    data: dict,
                    cache_dir: Path):
    file_path = cache_dir / f"{paper_id}.json"
    with open(file_path.with_suffix(".inprogress"), "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())

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
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        self.model, self.tokenizer, self.chunk_tokenizer = self.load_model()

        self.chunk_size = min(chunk_size, self.tokenizer.model_max_length//3
                              ) if chunk_size else self.tokenizer.model_max_length//3
        self.max_chunks = max_chunks
        self.chunk_batch_size = chunk_batch_size
        self.paper_batch_size = paper_batch_size
        self.cache_dir = config.TMP_PATH / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs

    def load_model(self):
        chunk_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if chunk_tokenizer.pad_token is None:
            chunk_tokenizer.pad_token = chunk_tokenizer.eos_token
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16).to(self.device).eval()
        return model, tokenizer, chunk_tokenizer

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

        for i in range(0, len(texts), self.chunk_batch_size):
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

    def batch_embeddings(self, paperData):
        papers = get_payload_and_text(paperData)
        papers = papers.to_dict(orient='records')

        save_executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        chunk_executor = ThreadPoolExecutor(max_workers=self.n_jobs)

        tqdm = get_tqdm()
        # with tqdm(range(0, len(papers), self.paper_batch_size),
        #           desc=f"Embedding {len(papers)} papers", position=0, colour='#000000') as pbar:
        # for i in pbar:
        paperBatch = papers  # [i:i+self.paper_batch_size]
        doc_prefix = self._model_prefix(tokenized=True)
        tasks = [chunk_executor.submit(
            load_and_chunk, paper, self.chunk_tokenizer,
            self.chunk_size, self.max_chunks, len(doc_prefix)) for paper in paperBatch]

        batched_texts = []
        batched_ids = []
        batched_masks = []
        batched_payloads = []
        all_paper_ids = []
        chunk_counts = []
        n_papers = len(tasks)
        with tqdm(enumerate(as_completed(tasks), 1), total=len(papers),
                  desc=f"Embedding {len(papers)} papers", position=0, colour='#000000') as pbar:
            for k, future in pbar:
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
                    print("Failed to chunk")
                    raise e

                pbar.set_postfix(
                    {'embedding': f"{n_papers - k + 1} papers left"})
                if (len(all_paper_ids) >= self.chunk_batch_size // self.max_chunks
                        or k == n_papers):
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
                        all_embeddings = torch.stack([emb for id, emb in zip(batched_ids, embeddings)
                                                      if id == paper_id])
                        mean_embedding = all_embeddings.mean(dim=0).tolist()
                        embeded_data = {
                            "embeddings": [
                                emb.tolist() for id, emb in zip(batched_ids, embeddings)
                                if id == paper_id],
                            "payloads": [
                                payload for id, payload in zip(batched_ids, batched_payloads)
                                if id == paper_id],
                            "mean_embedding": mean_embedding}
                        if embeded_data["embeddings"]:
                            save_executor.submit(
                                save_embeddings, paper_id, embeded_data, self.cache_dir)
                            # save_embeddings(paper_id, embeded_data, self.cache_dir)

                    batched_texts = []
                    batched_ids = []
                    batched_masks = []
                    batched_payloads = []
                    all_paper_ids = []
                    chunk_counts = []

        chunk_executor.shutdown(wait=True)
        save_executor.shutdown(wait=True)

    def embed_queries(self, queries):
        query_prefix = self._model_prefix(tokenized=False,
                                          role='query')
        embeddings = self.compute_embeddings(texts=queries,
                                             prefix=query_prefix)

        return embeddings
