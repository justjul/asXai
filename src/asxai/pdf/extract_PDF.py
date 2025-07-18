import os
import glob
from pathlib import Path
import shutil
import time
from datetime import datetime
from functools import partial

from typing import List, Optional, Any, TypedDict, Tuple, Union
import pickle
import pandas as pd
import math

import multiprocessing
from asxai.pdf.download_PDF import collect_downloaded_ids, downloaded_year_done

import config

import re
import numpy as np
from statistics import mode

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar, LTTextBox
from pdfminer.pdfpage import PDFPage

from asxai.utils import get_tqdm
from asxai.utils import load_params, save_parquet_dataset
import logging
from asxai.logger import get_logger

# Module-level logger
logger = get_logger(__name__, level=config.LOG_LEVEL)

# Suppress verbose pdfminer logs
pdfminer_logger = logging.getLogger('pdfminer')
pdfminer_logger.setLevel(logging.ERROR)

# Load parameters for PDF extraction
params = load_params()
pdf_config = params["pdf"]


class PaperInfo(TypedDict):
    """
    Holds minimal information for a PDF extraction task.

    Attributes:
        paperId (str): Unique paper identifier.
        openAccessPdf (str): URL or local path to the PDF file.
    """
    paperId: str
    openAccessPdf: str


class PDFextractor:
    """
    Monitors a download directory for completed PDFs downloads and extracts text blocks.

    Key steps:
    1. Wait for PDF files in per-paper subdirectories.
    2. Determine pages to parse based on word thresholds.
    3. Use pdfminer to split pages into text blocks with layout metadata.
    4. Serialize extraction output to pickle files.
    5. Optionally clean up original PDFs.
    """

    def __init__(
        self,
        download_dir: str,
        keepPDF: Optional[bool] = False,
        timeout: Optional[int] = 60,
        max_pages: Optional[Union[int, List[int]]] = None,
        n_word_th: Optional[int] = 50
    ):
        """
        Initialize PDFextractor.

        Args:
            download_dir: Base folder containing per-paper download subfolders.
            keepPDF: Whether to preserve PDF files after extraction.
            timeout: How long (s) to wait for a PDF to appear before skipping.
            max_pages: If list [from_start, from_end], limits pages considered.
            n_word_th: Minimum words for a page to be valid.
        """
        self.download_dir = download_dir
        self.keepPDF = keepPDF
        self.timeout = timeout
        self.max_pages = max_pages
        self.n_word_th = n_word_th

        # Create extraction output directory
        year_dir = Path(self.download_dir).stem
        self.extracted_dir = config.TMP_PATH / "extracted" / year_dir
        os.makedirs(self.extracted_dir, exist_ok=True)
        os.chmod(self.extracted_dir, 0o777)

    def extractPDFs(self, papers: List[PaperInfo]):
        """
        Process a batch of PaperInfo items, extracting each PDF when available.

        Args:
            papers: List of dicts containing 'paperId' and optional 'valid_pages'.
        Returns:
            paperIds that were processed.
        """
        # Track download directories and initial timestamps
        dir_list = [{"id": pdf_data["paperId"], "time": None}
                    for pdf_data in papers]
        while dir_list:
            for dirname in dir_list:
                dir_path = os.path.join(self.download_dir, dirname["id"])
                if os.path.isdir(dir_path):
                    if not dirname["time"]:
                        dirname["time"] = os.path.getmtime(dir_path)
                    if glob.glob(os.path.join(dir_path, "*.pdf")):
                        filepath = glob.glob(
                            os.path.join(dir_path, "*.pdf"))[0]
                        paper_data = next(
                            (pdf_data for pdf_data in papers if pdf_data["paperId"] == dirname["id"]), None)

                        if paper_data:
                            if ("valid_pages" not in paper_data
                                    or paper_data["valid_pages"] is None):
                                # paper_data["valid_pages"] = get_page_list(filepath,
                                #                                           max_pages=self.max_pages,
                                #                                           n_word_th=self.n_word_th)
                                if (self.max_pages is None or self.max_pages[1] > 0):
                                    paper_data["valid_pages"] = None
                                else:
                                    paper_data["valid_pages"] = list(
                                        range(self.max_pages[0]))

                            pdf_data = extract_pdf_sections(filepath,
                                                            valid_pages=paper_data["valid_pages"])
                            paper_data.pop("valid_pages")
                            for key in pdf_data.keys():
                                paper_data[key] = pdf_data[key]
                            paper_data["status"] = "extracted"

                            dic_path = Path(self.extracted_dir,
                                            dirname["id"] + ".pkl")

                            with open(dic_path.with_suffix(".inprogress"), "wb") as f:
                                pickle.dump(paper_data, f,
                                            protocol=pickle.HIGHEST_PROTOCOL)
                                f.flush()
                                os.fsync(f.fileno())

                            try:
                                dic_path.with_suffix(
                                    ".inprogress").rename(dic_path)
                            except Exception as e:
                                logger.warning(
                                    f"Couldn't rename {dic_path}: {e}")

                        dir_list.remove(dirname)
                        if not self.keepPDF and os.path.isdir(dir_path):
                            shutil.rmtree(dir_path, ignore_errors=True)
                    elif time.time() - dirname["time"] > self.timeout:
                        crfilepath = glob.glob(
                            os.path.join(dir_path, "*.crdownload"))
                        if not crfilepath or \
                                time.time() - os.path.getmtime(crfilepath[0]) > self.timeout:
                            dir_list.remove(dirname)
                            if not self.keepPDF and os.path.isdir(dir_path):
                                shutil.rmtree(dir_path, ignore_errors=True)

        return [pdf_data["paperId"] for pdf_data in papers]

    def push_Dummy_done(self):
        done_path = os.path.join(self.extracted_dir, "done.pkl")

        with open(done_path, "wb") as f:
            pickle.dump([0], f, protocol=pickle.HIGHEST_PROTOCOL)


def get_page_list(pdf_path: str,
                  n_word_th: int = 150,
                  max_pages: Optional[List[int]] = None) -> int:
    """Return total number of pages in a PDF."""
    # try:
    page_list = []
    with open(pdf_path, "rb") as f:
        try:
            all_pages = list(PDFPage.get_pages(f))
            n_pages = len(all_pages)
        except Exception:
            logger.warning(f"Could not open {pdf_path}")
            return -1

    page_lim = [n_pages, n_pages] if max_pages is None else max_pages

    for p in range(n_pages):
        if p < page_lim[0] or p >= n_pages - page_lim[1]:
            txt = extract_text(pdf_path, page_numbers=[p])
            nwords = len(
                [w for w in txt.split() if _is_normal_text(w)])
            if nwords > n_word_th:
                page_list.append(p)

    if not page_list:
        logger.warning(f"PDF {pdf_path} likely corrupted or too short")
        return []

    if max_pages:
        if len(page_list) > sum(max_pages):
            pages = page_list[:max_pages[0]]
            if max_pages[1] > 0:
                pages += page_list[-max_pages[1]:]
            page_list = pages

    return page_list
    # except Exception as e:
    #     logger.warning(f"Error reading PDF {pdf_path}: {e}")
    #     return None


def is_valid_page(pdf_path: str, page_idx: int) -> bool:
    """Check if a specific page in the PDF can be parsed without error."""

    try:
        for _ in extract_pages(pdf_path, page_numbers=[page_idx]):
            return True
    except Exception:
        return False


def get_valid_pages(
        pdf_path: str,
        timeout: float = 8.0,
        page_list: List[int] = None) -> List[int]:
    """Return a list of valid page indices."""

    valid_pages = []
    try:
        page_list = get_page_list(pdf_path) if page_list is None else page_list
        if page_list:
            n_pages = len(page_list)
            num_processes = min(20, n_pages)
            with multiprocessing.Pool(processes=num_processes) as pool:
                async_tasks = {pool.apply_async(is_valid_page, (pdf_path, p)): p
                               for p in page_list}
                pending_tasks = set(async_tasks.keys())
                endtime = time.time() + timeout
                while pending_tasks and time.time() < endtime:
                    for task in list(pending_tasks):
                        if task.ready():
                            if task.get():
                                valid_pages.append(async_tasks[task])
                                pending_tasks.remove(task)

                pool.terminate()
                pool.join()
                if len(valid_pages) < n_pages:
                    missing_pages = [
                        p for p in page_list if p not in valid_pages]
                    # logger.warning(
                    #     f"Timeout on page {missing_pages} out of {n_pages} for {pdf_path}")

    except Exception as e:
        logger.warning(f"Error reading PDF {pdf_path}: {e}")

    valid_pages.sort()
    return valid_pages


def _filter_text_block(element) -> Optional[Tuple[str, List[float]]]:
    filtered_text, sizes = [], []
    for text_line in element:
        if isinstance(text_line, LTTextLine):
            sizes.extend([
                character.size
                for character in text_line
                if isinstance(character, LTChar)])
            line_text = [character.get_text() for character in text_line]
            line_text = "".join(line_text).strip()
            if not line_text.isdigit():
                filtered_text.append(line_text)
            filtered_text.append("\n")

    cleaned_text = re.sub(r"\(cid:[^\)]+\)", "",
                          "".join(filtered_text).strip())
    return cleaned_text, sizes


ref_markers = {"references", "References",
               "REFERENCES", "Literature Cited"}


def _is_normal_text(word):
    if any(ch.isdigit() for ch in word):
        return False
    if any(ch.isupper() for ch in word):
        return False
    return True


def _extract_blocks_pdfminer(
        pdf_path: str,
        valid_pages: Optional[List[int]] = None) -> dict:

    sections = {"full_text": None,
                "main_text": None,
                "ref_text": None}
    if valid_pages is None:
        valid_pages = get_page_list(pdf_path,
                                    max_pages=None,
                                    n_word_th=0)
    if not valid_pages:
        return sections
    # Determine the most common font size (mode)
    page_width, page_height = 595, 842
    all_text_blocks, all_fontsize_blocks = [], []
    all_page_blocks = []
    all_Bottom_blocks, all_Top_blocks = [], []
    all_Left_blocks, all_Right_blocks = [], []
    all_Bottom_words, all_Top_words = [], []
    all_Left_words, all_Right_words = [], []
    all_nwords_blocks = []

    try:
        for p, page_layout in enumerate(extract_pages(pdf_path, page_numbers=valid_pages)):
            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue

                left, bottom, right, top = element.bbox
                left /= page_width
                right = 1 - right/page_width
                bottom /= page_height
                top = 1 - top/page_height

                filtered_text, fontsizes = _filter_text_block(element)
                letter_count = 0
                for c in filtered_text:
                    letter_count += c.isalpha()
                    if letter_count > 5:
                        break
                if letter_count > 5:
                    all_text_blocks.append(filtered_text)
                    all_fontsize_blocks.append(mode(fontsizes))
                    all_page_blocks.append(valid_pages[p])
                    nwords = len(filtered_text.split(' '))
                    all_nwords_blocks.append(nwords)
                    all_Bottom_blocks.append(bottom)
                    all_Top_blocks.append(top)
                    all_Left_blocks.append(left)
                    all_Right_blocks.append(right)

                    all_Bottom_words.extend([bottom]*nwords)
                    all_Top_words.extend([top]*nwords)
                    all_Left_words.extend([left]*nwords)
                    all_Right_words.extend([right]*nwords)

        q_th = 0.02
        nwords_th = 10

        bottom_th = np.quantile(all_Bottom_words, q_th)
        top_th = np.quantile(all_Top_words, q_th)
        left_th = np.quantile(all_Left_words, q_th)
        right_th = np.quantile(all_Right_words, q_th)

        blocks = [block for i, block in enumerate(zip(all_text_blocks, all_fontsize_blocks,
                                                      all_page_blocks,
                                                      all_Bottom_blocks, all_Top_blocks,
                                                      all_Left_blocks, all_Right_blocks,))
                  if ((all_Bottom_blocks[i] > bottom_th and
                       all_Top_blocks[i] > top_th and
                       all_Left_blocks[i] > left_th and
                       all_Right_blocks[i] > right_th) or
                      (all_nwords_blocks[i] > nwords_th))]

        sections["full_text"] = ''.join(
            f"\n**BLOCK**"
            f"fs=={size: .1f}**"
            f"p=={page: .1f}**"
            f"b=={bottom: .1f}**"
            f"t=={top: .1f}**"
            f"l=={left: .1f}**"
            f"r=={right: .1f}**"
            f"\n{text}"
            for (text, size, page, bottom, top, left, right) in blocks)
    except Exception as e:
        pass  # logger.warning(f"Issue when parsing {pdf_path}: {e}")

    return sections


def extract_pdf_sections(
        pdf_path: str,
        engine: str = "pdfminer",
        valid_pages: List[int] = None) -> dict:

    if "pdfminer" in engine.lower():
        sections = _extract_blocks_pdfminer(pdf_path, valid_pages)
        return sections
    else:
        raise Exception("No other option than pdfminer implemented yet")


def get_valid_pages_PDFs(
        papers: List[PaperInfo],
        directory: str,
        timeout: int = 8.0,
        page_range: List[int] = None):

    # Extracting pdfs from downloads that took too long to be processed immediately
    # downloads_dir = "/tmp/s2tmp"
    if os.path.isdir(directory):
        dir_list = os.listdir(directory)
        dir_list.sort(key=lambda name: os.path.getmtime(
            os.path.join(directory, name)))
        for dirname in dir_list:
            dir_path = os.path.join(directory, dirname)
            if os.path.isdir(dir_path):
                if glob.glob(os.path.join(dir_path, "*.pdf")):
                    filepath = glob.glob(os.path.join(dir_path, "*.pdf"))[0]
                    paper_data = next((pdf_data
                                       for pdf_data in papers
                                       if pdf_data["paperId"] == dirname), None)

                    if paper_data:
                        if page_range is None:
                            page_list = get_page_list(filepath)
                        else:
                            page_list = list(
                                range(page_range[0], page_range[1]))
                        valid_pages = get_valid_pages(filepath,
                                                      timeout=timeout,
                                                      page_list=page_list)

                        paper_data["valid_pages"] = valid_pages
    return papers


def get_unprocessed_PDFs(
        papers: List[PaperInfo],
        directory: str):

    filtered_papers = []
    if os.path.isdir(directory):
        filtered_papers = []
        for paper in papers:
            try:
                pdf_path = os.path.join(directory, paper["paperId"])
                dic_path = os.path.join(directory,
                                        "extracted",
                                        paper["paperId"] + ".pkl")

                if os.path.isdir(pdf_path) and not os.path.isfile(dic_path):
                    if glob.glob(os.path.join(pdf_path, "*.pdf")):
                        filtered_papers.append(paper)
            except Exception:
                pass

    return filtered_papers


def extraction_year_done(directory: str):
    dic_list = glob.glob(os.path.join(directory, "*.pkl"))
    return 'done' in [Path(f).stem for f in dic_list]


def collect_extracted_ids(directory: str):
    dic_list = glob.glob(os.path.join(directory, "*.pkl"))
    return [Path(f).stem for f in dic_list if Path(f).stem != 'done']


def collect_extracted_PDFs(directory: str, paperIds: List[str], batch_size: int = 32):
    dic_list = sorted(
        glob.glob(os.path.join(directory, "*.pkl")),
        key=os.path.getmtime
    )
    dic_list = dic_list[:batch_size]
    final_results = []
    for paper_path in dic_list:
        if os.path.isfile(paper_path) and Path(paper_path).stem in paperIds:
            try:
                with open(paper_path, "rb") as f:
                    final_results.append(pickle.load(f))
                if os.path.isfile(paper_path):
                    try:
                        os.remove(paper_path)
                    except Exception:
                        logger.exception(f"Unable to remove {paper_path}")
            except Exception as e:
                print(e)

    return final_results


def extract_worker_init(
        downloads_dir: str,
        keepPDF: Optional[bool] = False,
        timeout: Optional[float] = 60,
        max_pages: Optional[Union[int, List[int]]] = None):
    global pdfextractor
    pdfextractor = PDFextractor(downloads_dir,
                                keepPDF=keepPDF,
                                timeout=timeout,
                                max_pages=max_pages)


def extract_PDFs_workers(papers):
    global pdfextractor
    output = pdfextractor.extractPDFs(papers)
    return output


def extracted_to_text(data, year):
    output_dir_year = config.TMP_PATH / "text_to_save" / str(year)
    os.makedirs(output_dir_year, exist_ok=True)
    os.chmod(output_dir_year, 0o777)
    filepath = output_dir_year / f"text_{year}"

    save_parquet_dataset(data, output_dir=filepath, compression="snappy")


def extracted_to_DB(textdata, metadata):
    assert textdata['paperId'].equals(metadata['paperId'])
    id0 = textdata['paperId'].iloc[0]
    output_dir_DB = config.TMP_PATH / "text_to_embed" / id0
    os.makedirs(output_dir_DB, exist_ok=False)
    os.chmod(output_dir_DB, 0o777)

    fp_text = output_dir_DB / "text.inprogress"
    save_parquet_dataset(textdata, output_dir=fp_text, compression="snappy")
    fp_text.rename(output_dir_DB / "text")

    fp_meta = output_dir_DB / "metadata.inprogress"
    save_parquet_dataset(metadata, output_dir=fp_meta, compression="snappy")
    fp_meta.rename(output_dir_DB / "metadata")


def collect_extracted_batch(directory: Path):
    if not directory.is_dir():
        return None

    extracted_batches = []
    for batch_path in directory.iterdir():
        if batch_path.is_dir():
            text_dir = batch_path / "text"
            metadata_dir = batch_path / "metadata"
            if text_dir.is_dir() and metadata_dir.is_dir():
                extracted_batches.append(batch_path.name)

    return extracted_batches


def batch_full_extract(paperInfo,
                       downloads_dir,
                       n_jobs: Optional[int] = pdf_config['n_jobs_extract'],
                       timeout_per_article: Optional[float] = pdf_config['timeout_per_article'],
                       max_pages: Optional[Union[int, List[int]]] = [
                           pdf_config['max_pages_start'], pdf_config['max_pages_end']],
                       keep_pdfs: Optional[bool] = pdf_config['keep_pdfs']):

    minibatch_size = 1  # 100
    batch_size_base = 10 * n_jobs
    time_out_list = [1 * timeout_per_article,
                     2 * timeout_per_article,
                     0]
    timeout_iterator = (s for s in time_out_list)
    new_ids = collect_downloaded_ids(downloads_dir)
    extracted_ids, pending_ids = [], []
    year_ids = set(paperInfo["paperId"])
    endtime = time.time() + timeout_per_article * \
        math.ceil(len(paperInfo) // (batch_size_base + 1))
    while True:
        downloaded_ids = collect_downloaded_ids(downloads_dir)
        paper_ids = [id for id in downloaded_ids if id in year_ids]

        new_ids = [id for id in paper_ids if (id not in extracted_ids
                                              and id not in pending_ids)]

        if not new_ids:
            if pending_ids:
                new_ids = pending_ids
                pending_ids = []
                batch_size = len(new_ids)
                logger.info(
                    f"Will now extract {len(new_ids)} papers that were pending")
            elif downloaded_year_done(downloads_dir):
                break
            else:
                time.sleep(5)
                logger.info(
                    f"no new data. Will wait {endtime - time.time():.1f} more seconds")
                continue
        else:
            batch_size = batch_size_base
            timeout_iterator = (s for s in time_out_list)

        time_out_base = next(timeout_iterator)
        time_out = time_out_base * \
            math.ceil(min(len(new_ids), batch_size) / n_jobs)
        do_pending_papers = False if time_out else True

        paperInfo_new = paperInfo[paperInfo["paperId"].isin(new_ids)]
        Npapers = len(paperInfo_new)
        tqdm = get_tqdm()
        with tqdm(range(math.ceil(Npapers / (batch_size + 1))),
                  desc=f"Extracting {Npapers} new papers") as pbar:
            try:
                for i in pbar:
                    batch = paperInfo_new.iloc[i *
                                               batch_size: (i+1)*batch_size].to_dict(orient="records")

                    Nbatch = len(batch)
                    minibatches = [batch[k * minibatch_size: (k + 1) * minibatch_size]
                                   for k in range(Nbatch // minibatch_size + 1)]

                    minibatches = [b for b in minibatches
                                   if len(b) > 0 and b[0] is not None]

                    Nminibatches = len(minibatches)
                    if minibatches:
                        extract_pool = multiprocessing.Pool(
                            processes=min(Nminibatches, n_jobs),
                            initializer=partial(
                                extract_worker_init,
                                downloads_dir=downloads_dir,
                                keepPDF=keep_pdfs,
                                timeout=60,
                                max_pages=max_pages))

                        with extract_pool:
                            async_extract = [extract_pool.apply_async(extract_PDFs_workers, (b,))
                                             for b in minibatches]

                            pending_extract = set(async_extract)

                            end_time_main = time.time() + time_out
                            end_time = end_time_main
                            while pending_extract:
                                if time.time() > end_time:
                                    logger.info(
                                        f"{len(pending_extract)} extractions out of {Nminibatches} timed out in {time_out:.1f}s. Terminating the pool.")

                                    extract_pool.terminate()
                                    extract_pool.join()
                                    break

                                for task in list(pending_extract):
                                    if task.ready():
                                        pending_extract.remove(task)
                                        extracted_ids.extend(task.get())

                                pbar.set_postfix(
                                    {"extraction tasks": f"{len(pending_extract)} left"})

                                time.sleep(1)

                                if (end_time == end_time_main
                                        and len(pending_extract) <= n_jobs):
                                    end_time = time.time() + time_out_base

            except Exception as e:
                pbar.close()
                raise e

        if do_pending_papers:
            pending_papers = paperInfo[paperInfo["paperId"].isin(
                new_ids)].to_dict(orient="records")

            logger.info(f"checking valid pages for {len(pending_papers)}" +
                        "articles that timed out...")

            pmax = max_pages[0] if max_pages else 20
            get_valid_pages_PDFs(papers=pending_papers,
                                 directory=downloads_dir,
                                 page_range=[0, pmax],
                                 timeout=0.4)

            n_pending = len(pending_papers)
            minib_size = math.ceil(
                len(pending_papers) / (n_jobs + 1))

            pending_minibatches = [pending_papers[k * minib_size: (k + 1) * minib_size]
                                   for k in range(n_jobs + 1)]

            pending_minibatches = [b for b in pending_minibatches
                                   if len(b) > 0 and b[0] is not None]

            n_pending = len(pending_minibatches)

            with multiprocessing.Pool(
                    processes=min(n_pending, n_jobs),
                    initializer=partial(
                        extract_worker_init,
                        downloads_dir=downloads_dir,
                        keepPDF=keep_pdfs,
                        timeout=60,
                        max_pages=max_pages
                    )) as extract_pool:

                async_extract = [extract_pool.apply_async(extract_PDFs_workers, (b,))
                                 for b in pending_minibatches]

                pending = set(async_extract)
                extratime = timeout_per_article * \
                    math.ceil(n_pending / n_jobs)
                end_time_extra = time.time() + extratime

                pbar.set_postfix(
                    {"status": f"will process now {n_pending} articles ( <{extratime} s)"})

                pending_ids_done = []
                while pending:
                    if time.time() > end_time_extra:
                        logger.info(f"{len(pending)} tasks out of {n_pending}" +
                                    "timed out after checking valid pdf pages.")

                        extract_pool.terminate()
                        extract_pool.join()
                        break

                    for task in list(pending):
                        if task.ready():
                            pending_ids_done.extend(task.get())
                            pending.remove(task)
                    if pending:
                        time.sleep(0.5)

            logger.info(
                f"Successfully extracted {len(pending_ids_done)} pending papers")
            extracted_ids.extend(new_ids)

        pending_ids.extend([id for id in new_ids if id not in extracted_ids])
        logger.info(f"{len(pending_ids)} papers on pending list")

    PDFextractor(downloads_dir).push_Dummy_done()


def _get_ref_index(blocks, fs_normal_text):
    year_markers = {str(y) for y in range(1950, datetime.now().year)}
    ref_idx_start, ref_idx_end = None, None
    Nbnext = 20
    for i, block in enumerate(blocks):
        if block['content']:
            heading_top = ' '.join(
                block['content'].splitlines()[0].split(' ')[:2])
            is_ref_top = any(mark in heading_top for mark in ref_markers)
            heading_bottom = ' '.join(
                block['content'].splitlines()[-1].split(' ')[:2])
            is_ref_bottom = any(mark in heading_bottom for mark in ref_markers)
            if is_ref_top or is_ref_bottom:
                Nbnext_real = min(Nbnext, len(blocks)-i)
                is_small_fonts = all(bnext['fontsize'] < fs_normal_text
                                     for bnext in blocks[i+1:i+1 + Nbnext])
                is_digit_start = sum(any(ch.isdigit()
                                         for ch in bnext['content'].replace('\n', '')[0:3])
                                     for bnext in blocks[i+1:i+1 + Nbnext]) > Nbnext_real // 2
                is_year_in = sum(any(y in bnext['content']
                                     for y in year_markers)
                                 for bnext in blocks[i:i + Nbnext]) > Nbnext_real // 3
                if is_small_fonts or is_digit_start or is_year_in:
                    if is_ref_top:
                        ref_idx_start = i
                        break
                    if is_ref_bottom:
                        ref_idx_start = i + 1
                        break

    if ref_idx_start is None:
        for i, block in enumerate(blocks):
            if block['content']:
                Nbnext_real = min(Nbnext, len(blocks)-i)
                is_small_fonts = all(bnext['fontsize'] < fs_normal_text
                                     for bnext in blocks[i:i + Nbnext])
                is_digit_start = sum(any(ch.isdigit()
                                         for ch in bnext['content'].replace('\n', '')[0:3])
                                     for bnext in blocks[i:i + Nbnext]) > Nbnext_real // 2
                if is_small_fonts or is_digit_start:
                    is_year_in = sum(any(y in bnext['content']
                                     for y in year_markers)
                                     for bnext in blocks[i:i + Nbnext]) > Nbnext_real // 3
                    is_punc_in = sum({'.', ','}.issubset(bnext['content'])
                                     for bnext in blocks[i:i + Nbnext]) > Nbnext_real // 2
                    if is_year_in and is_punc_in:
                        ref_idx_start = i
                        break
    if ref_idx_start is not None:
        for i, block in enumerate(blocks[ref_idx_start:-1]):
            if block['content']:
                if is_small_fonts:
                    if (block['fontsize'] >= fs_normal_text
                            and block['nwords'] >= 5):
                        ref_idx_end = ref_idx_start + i
                        break
                    else:
                        ref_idx_end = len(blocks)
        if ref_idx_end is None:
            ref_idx_end = len(blocks)
    return ref_idx_start, ref_idx_end


# BLOCK_pattern = re.compile(
#     r'(?s)\*\*BLOCK\*\*fs==\s*([\d.]+)\*\*p==\s*([\d.]+)\*\*b==\s*([\d.]+)\*\*t==\s*([\d.]+)\*\*l==\s*([\d.]+)\*\*r==\s*([\d.]+)\*\*(.*?)(?=\*\*BLOCK\*\*fs==|$)')


def _get_block_text_specs(text: str,
                          extract_ref: bool = False):
    blocks = []
    all_fs, all_p = [], []
    all_b, all_t, all_l, all_r = [], [], [], []
    n_words = []

    chunks = text.split("**BLOCK**fs==")
    blocks = []
    for chunk in chunks[1:]:
        try:
            parts, content = chunk.split("**", 6), chunk.split("**", 7)[-1]
            fs_val, p_val, b_val, t_val, l_val, r_val = map(
                float, [s.split('==')[-1] for s in parts[:6]]
            )
            words = content.strip().split(" ")

            nw = len([w for w in words if _is_normal_text(w)])
            n_words.append(nw)

            all_fs += [fs_val]*nw
            all_p += [p_val]*nw
            all_b += [b_val]*nw
            all_t += [t_val]*nw
            all_l += [l_val]*nw
            all_r += [r_val]*nw

            blocks.append({'fontsize': fs_val, 'page': p_val, 'bottom': b_val, 'top': t_val,
                           'left': l_val, 'right': r_val, 'nwords': nw,
                           'content': content, 'possible_ref': False})
        except Exception as e:
            logger.warning(f"Parsing of text blocks failed: {e}")
            continue

    all_fs_ref = []
    if extract_ref:
        try:
            fs_normal_text = mode([fs for i, fs in enumerate(all_fs)
                                   if all_p[i] <= max(all_p) // 2])
        except Exception:
            logger.warning(f"issue when estimating font size: {len(all_fs)}")
            if len(all_fs) > 5:
                fs_normal_text = mode(all_fs)
            else:
                fs_normal_text = 0

        ref_idx_start, ref_idx_end = _get_ref_index(blocks, fs_normal_text)

        if ref_idx_start:
            for block in blocks[ref_idx_start:ref_idx_end]:
                all_fs_ref.extend([block['fontsize']])
                block['possible_ref'] = True

    specs = {'fontsize': all_fs, 'page': all_p, 'bottom': all_b, 'top': all_t,
             'left': all_l, 'right': all_r, 'nwords': n_words,
             'fontsize_ref': all_fs_ref}

    return blocks, specs


def get_clean_block_text(text: str,
                         extract_ref: bool = False):
    if not text:
        return None, None

    blocks, specs = _get_block_text_specs(text, extract_ref=extract_ref)

    if not blocks:
        return "", ""

    try:
        fs_vals = specs.get("fontsize", [])
        if not fs_vals or len(set(fs_vals)) > 1000:
            logger.warning(
                "Unrealistic font size distribution — skipping mode calculation")
            fs_normal_text = 0
        else:
            fs_normal_text = mode([fs for i, fs in enumerate(specs['fontsize'])
                                   if specs['page'][i] <= max(specs['page']) // 2])
    except Exception:
        logger.warning("There's been an issue during font size calculation")
        if len(specs['fontsize_ref']) > 5:
            fs_normal_text = mode(specs['fontsize'])
        else:
            fs_normal_text = 0

    inblocks = [blk['content'] for blk in blocks
                if (blk['fontsize'] >= 0.95*fs_normal_text and blk['nwords'] > 0)]
    main_text = ' '.join(inblocks)

    main_text = ' '.join(main_text.splitlines())

    # Reference section should be split according to dates
    # Embeddings of these chunks should be then averaged
    # to give a single embedding per paper.
    # If no ref section is present, replace it with ref titles from s2
    # If absent in s2, replace it with average embeddings of chunks
    ref_text = None
    if len(specs['fontsize_ref']) > 5:
        fs_ref_text = mode(
            specs['fontsize_ref'][:len(specs['fontsize_ref'])//2])

        refblocks = [blk['content'] for blk in blocks
                     if (blk['fontsize'] == fs_ref_text and blk['possible_ref'])]
        ref_text = ' '.join(refblocks)
        ref_text = ' '.join(ref_text.splitlines())
        for mark in ref_markers:
            ref_text = ref_text.replace(mark, '')
            ref_text = re.sub(
                r'\b([A-Z][a-z]*|[A-Z])\.', r'\1', ref_text)

    return main_text, ref_text


def clean_full_text(pdf, extract_ref):
    try:
        full_text = pdf.get("full_text", "")
        if not isinstance(full_text, str) or len(full_text) < 100:
            logger.warning("Skipping overly short or malformed full_text")
            pdf["main_text"], pdf["ref_text"] = "", ""
        else:
            pdf["main_text"], pdf["ref_text"] = get_clean_block_text(
                full_text, extract_ref)
    except Exception as e:
        logger.warning(f"Issue cleaning PDF {pdf.get('paperId', '?')}: {e}")
        pdf["main_text"], pdf["ref_text"] = "", ""
    return pdf


def batch_full_Clean(extracted_dir: str,
                     paperdata: pd.DataFrame,
                     year: int,
                     extract_ref: Optional[bool] = pdf_config['extract_ref'],
                     n_jobs: Optional[int] = pdf_config['n_jobs_extract'],
                     push_to_vectorDB: Optional[bool] = False):
    ids_to_save = []
    while True:
        try:
            new_ids = collect_extracted_ids(extracted_dir)
            if not new_ids:
                if extraction_year_done(extracted_dir):
                    logger.warning("EXTRACTION DONE SIGNAL DETECTED")
                    break
                else:
                    logger.warning(
                        "Waiting: no new files yet, extract not done")
                    time.sleep(5)
                    continue

            batch_size = 64
            extracted_pdfs = collect_extracted_PDFs(
                extracted_dir, new_ids, batch_size=batch_size)
        except Exception:
            logger.exception("Cleanup failed during pdf collection")
            raise

        logger.info("Still running clean loop…")
        Npapers = len(extracted_pdfs)
        tqdm = get_tqdm()
        with tqdm(range(math.ceil(Npapers / (batch_size + 1))),
                  desc=f"Cleaning {Npapers} new papers") as pbar:
            try:
                for i in pbar:
                    batch_pdfs = extracted_pdfs[i*batch_size: (i+1)*batch_size]
                    with multiprocessing.Pool(processes=n_jobs) as clean_pool:
                        clean_async = [clean_pool.apply_async(clean_full_text, (pdf, extract_ref))
                                       for pdf in batch_pdfs]

                        pending = set(clean_async)
                        final_results = []

                        batch_timeout = 60 * \
                            math.ceil(len(batch_pdfs) / n_jobs)
                        end_time = time.time() + batch_timeout

                        while pending:
                            if time.time() > end_time:
                                logger.warning(
                                    f"{len(pending)} out of {len(batch_pdfs)} cleaning tasks timed out. Terminating pool.")
                                clean_pool.terminate()
                                clean_pool.join()
                                break

                            for task in list(pending):
                                if task.ready():
                                    try:
                                        result = task.get()
                                        final_results.append(result)
                                    except Exception as e:
                                        logger.warning(
                                            f"Cleaning task failed: {e}")
                                    pending.remove(task)

                            pbar.set_postfix(
                                {"cleaning tasks": f"{len(pending)} left"})

                            time.sleep(1)
            except Exception as e:
                pbar.close()
                raise e

        if final_results:
            try:
                pdfdata = pd.DataFrame(
                    [pdf for pdf in final_results])

                ids_to_save.extend(pdfdata["paperId"])

                # Ensure unique DOI index before combining
                pdfdata_clean = pdfdata.drop_duplicates(
                    subset="doi", keep="first")
                text_clean = paperdata["text"].drop_duplicates(
                    subset="doi", keep="first")

                # Merge using DOI
                paperdata_text = (
                    pdfdata_clean.set_index("doi")
                    .combine_first(text_clean.set_index("doi"))
                    .reset_index(drop=False)
                )

                # Re-align to match original paperId order in metadata
                # by joining on 'paperId'
                paperdata["text"] = paperdata["metadata"][["paperId"]].merge(
                    paperdata_text, on="paperId", how="left"
                )

                # paperdata["text"] = (pdfdata.set_index("doi").combine_first(
                #     paperdata["text"].set_index("doi")).reset_index(drop=False))

                paperdata["text"]["authorName"] = paperdata["metadata"]["authorName"]
            except:
                logger.exception("Cleanup failed during assembling results")
                raise

        if len(ids_to_save) > 64:
            try:
                extracted_to_text(paperdata["text"], year)

                if push_to_vectorDB:
                    DBmask = paperdata['text']["paperId"].isin(
                        ids_to_save)
                    text_to_DB = paperdata['text'].loc[DBmask]
                    metadata_to_DB = paperdata['metadata'].loc[DBmask]
                    extracted_to_DB(text_to_DB, metadata_to_DB)
            except:
                logger.exception("Cleanup failed during saving")
                raise

            ids_to_save = []

    if len(ids_to_save) > 0:
        extracted_to_text(paperdata["text"], year)

        if push_to_vectorDB:
            DBmask = paperdata['text']["paperId"].isin(
                ids_to_save)
            text_to_DB = paperdata['text'].loc[DBmask]
            metadata_to_DB = paperdata['metadata'].loc[DBmask]
            extracted_to_DB(text_to_DB, metadata_to_DB)

        ids_to_save = []

    if os.path.isdir(extracted_dir):
        shutil.rmtree(extracted_dir, ignore_errors=True)


def extract_PDFs(
        paperdata: pd.DataFrame,
        year: int,
        n_jobs: Optional[int] = pdf_config['n_jobs_extract'],
        timeout_per_article: Optional[float] = pdf_config['timeout_per_article'],
        max_pages: Optional[Union[int, List[int]]] = [
            pdf_config['max_pages_start'], pdf_config['max_pages_end']],
        extract_ref: Optional[bool] = pdf_config['extract_ref'],
        pdfs_dir: Optional[str | Path] = pdf_config['save_pdfs_to'],
        keep_pdfs: Optional[bool] = pdf_config['keep_pdfs'],
        push_to_vectorDB: Optional[bool] = False,
        extract_done: Optional[bool] = None):

    class DummyEvent:
        def is_set(self): return True
        def set(self): return True
    if extract_done is None:
        extract_done = DummyEvent()

    if pdfs_dir is not None:
        downloads_dir_base = pdfs_dir
    else:
        downloads_dir_base = config.TMP_PATH / "downloads"

    if not isinstance(max_pages, list):
        max_pages = [max_pages, 0]
    max_pages = [n or 0 for n in max_pages]
    if not any(max_pages):
        max_pages = None

    n_jobs = min(n_jobs, 2 * multiprocessing.cpu_count() // 3)

    downloads_dir = downloads_dir_base / str(year)
    extracted_dir = config.TMP_PATH / "extracted" / str(year)
    os.makedirs(extracted_dir, exist_ok=True)
    os.chmod(extracted_dir, 0o777)

    logger.info(f"Extracting pdfs for year {year}")
    paperInfo = pd.merge(paperdata["text"][["doi", "paperId", "openAccessPdf"]],
                         paperdata["metadata"][["doi", "authorName"]],
                         on="doi",
                         how="inner")

    extract_process = multiprocessing.Process(
        target=batch_full_extract,
        kwargs={'paperInfo': paperInfo,
                'downloads_dir': downloads_dir,
                'n_jobs': n_jobs,
                'timeout_per_article': timeout_per_article,
                'max_pages': max_pages,
                'keep_pdfs': keep_pdfs})

    clean_process = multiprocessing.Process(
        target=batch_full_Clean,
        kwargs={'extracted_dir': extracted_dir,
                'paperdata': paperdata,
                'year': year,
                'extract_ref': extract_ref,
                'n_jobs': n_jobs,
                'push_to_vectorDB': push_to_vectorDB})

    extract_process.start()
    clean_process.start()

    clean_process.join()
    extract_process.join()

    if not keep_pdfs:
        try:
            if os.path.isdir(downloads_dir):
                shutil.rmtree(downloads_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(
                f"Could not delete downloads folder for {year}: {e}")

    extract_done.set()
