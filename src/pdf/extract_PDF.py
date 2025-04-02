import os
import glob
from pathlib import Path
import shutil
import time
from datetime import datetime
from functools import partial

from typing import List, Optional, Any, TypedDict, Tuple
import pickle
import pandas as pd
import math

import multiprocessing
from dataIO.load import load_data
from pdf.download_PDF import collect_downloaded_ids

import config

import re
import numpy as np
from statistics import mode

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar, LTTextBox
from pdfminer.pdfpage import PDFPage

from src.utils import get_tqdm
from src.utils import load_params
import logging
from src.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)
pdfminer_logger = logging.getLogger('pdfminer')
pdfminer_logger.setLevel(logging.ERROR)

params = load_params()
pdf_config = params["pdf"]


class PaperInfo(TypedDict):
    paperId: str
    openAccessPdf: str


class PDFextractor:
    def __init__(
            self,
            directory: str,
            keepPDF: Optional[bool] = False,
            timeout: Optional[int] = 60):

        self.directory = directory
        self.keepPDF = keepPDF
        self.timeout = timeout
        self.processed_dir = os.path.join(self.directory, "extracted")
        os.makedirs(self.processed_dir, exist_ok=True)

    def extractPDFs(self, papers: List[PaperInfo]):
        # dir_path_orig = [os.path.join(downloads_dir, pdf_data['paperId']) for pdf_data in textdata]
        dir_list = [{"id": pdf_data["paperId"], "time": None}
                    for pdf_data in papers]
        while dir_list:
            # if os.path.isdir(downloads_dir):
            #     dir_list = os.listdir(downloads_dir) if dir_list is None else dir_list
            # dir_list.sort(key=lambda name: os.path.getmtime(os.path.join(downloads_dir, name)))
            for dirname in dir_list:
                dir_path = os.path.join(self.directory, dirname["id"])
                if os.path.isdir(dir_path):
                    if not dirname["time"]:
                        dirname["time"] = os.path.getmtime(dir_path)
                    if glob.glob(os.path.join(dir_path, "*.pdf")):
                        filepath = glob.glob(
                            os.path.join(dir_path, "*.pdf"))[0]
                        paper_data = next(
                            (pdf_data for pdf_data in papers if pdf_data["paperId"] == dirname["id"]), None)

                        if paper_data:
                            if "valid_pages" not in paper_data.keys():
                                paper_data["valid_pages"] = None
                            pdf_data = extract_pdf_sections(filepath,
                                                            valid_pages=paper_data["valid_pages"])

                            for key in pdf_data.keys():
                                paper_data[key] = pdf_data[key]
                            paper_data["pdf_status"] = "extracted"

                            dic_path = os.path.join(self.processed_dir,
                                                    dirname["id"] + ".pkl")

                            with open(dic_path, "wb") as f:
                                pickle.dump(paper_data, f,
                                            protocol=pickle.HIGHEST_PROTOCOL)

                        dir_list.remove(dirname)
                        if not self.keepPDF:
                            shutil.rmtree(dir_path)
                    elif time.time() - dirname["time"] > self.timeout:
                        crfilepath = glob.glob(
                            os.path.join(dir_path, "*.crdownload"))
                        if not crfilepath or \
                                time.time() - os.path.getmtime(crfilepath[0]) > self.timeout:
                            dir_list.remove(dirname)
                            if not self.keepPDF:
                                shutil.rmtree(dir_path)
        return papers


def get_pdf_num_pages(pdf_path: str) -> int:
    """Return total number of pages in a PDF."""
    try:
        with open(pdf_path, "rb") as f:
            return sum(1 for _ in PDFPage.get_pages(f))
    except Exception as e:
        logger.warning(f"Error reading PDF {pdf_path}: {e}")
        return None


def is_valid_page(pdf_path: str, page_idx: int) -> bool:
    """Check if a specific page in the PDF can be parsed without error."""

    try:
        for _ in extract_pages(pdf_path, page_numbers=[page_idx]):
            return True
    except Exception:
        return False


def get_valid_pages(
        pdf_path: str,
        timeout: float = 1.0,
        num_processes: int = 1) -> List[int]:
    """Return a list of valid page indices."""

    valid_pages = []
    try:
        n_pages = get_pdf_num_pages(pdf_path)
        with multiprocessing.Pool(processes=num_processes) as pool:
            async_results = [pool.apply_async(is_valid_page, (pdf_path, i))
                             for i in range(n_pages)]
            for i, result in enumerate(async_results):
                try:
                    if result.get(timeout=timeout):
                        valid_pages.append(i)
                except multiprocessing.TimeoutError:
                    logger.warning(
                        f"Timeout on page {i} out of {n_pages} for {pdf_path}")
                    pool.terminate()
                    break
    except Exception as e:
        logger.warning(f"Error reading PDF {pdf_path}: {e}")

    return valid_pages


def pdf_main_fontsize(
        pdf_path: str,
        valid_pages: Optional[List[int]] = None) -> Optional[float]:
    """Estimate the main font size used in the document."""

    try:
        sizes = [
            character.size
            for page_layout in extract_pages(pdf_path, page_numbers=valid_pages)
            for element in page_layout
            if isinstance(element, LTTextContainer)
            for text_line in element
            if isinstance(text_line, LTTextLine)
            for character in text_line
            if isinstance(character, LTChar)]
        return mode(sizes) if sizes else None
    except Exception as e:
        logger.warning(f"Font size extraction failed for {pdf_path}: {e}")
        return None


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


def _get_markers():
    last_section_markers = {
        "acknowledgment",
        "acknowledgement",
        "acknowlegment",
        "reference",
    }
    month_markers = {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "nov",
        "dec",
    }
    year_markers = {str(y) for y in range(1900, 2030)}
    section_list = {
        "introduction",
        "results",
        "discussion",
        "conclusions",
        "methods",
        "materials",
        "experimental",
        "materials and methods",
        "experimental procedure",
        "related work",
        "i.",
        "ii.",
        "iii.",
        "iv.",
        "v.",
        "vi.",
    }
    return last_section_markers, month_markers, year_markers, section_list


BLOCK_pattern = re.compile(
    r'(?s)\*\*BLOCK\*\*fs==\s*([\d.]+)\*\*b==\s*([\d.]+)\*\*t==\s*([\d.]+)\*\*l==\s*([\d.]+)\*\*r==\s*([\d.]+)\*\*(.*?)(?=\*\*BLOCK\*\*fs==|$)')

ref_markers = {"references", "References",
               "REFERENCES", "Literature Cited"}


def _is_normal_text(word):
    if any(ch.isdigit() for ch in word):
        return False
    if any(ch.isupper() for ch in word):
        return False
    return True


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


def _get_block_text_specs(text):
    blocks = []
    all_fs, all_b, all_t, all_l, all_r = [], [], [], [], []
    n_words = []
    for match in BLOCK_pattern.finditer(text):
        fs_val = float(match.group(1))
        b_val = float(match.group(2))
        t_val = float(match.group(3))
        l_val = float(match.group(4))
        r_val = float(match.group(5))
        content = match.group(6).strip()
        words = content.split(' ')

        nw = len([w for w in words if _is_normal_text(w)])
        n_words.append(nw)

        all_fs.extend([fs_val]*nw)
        all_b.extend([b_val]*nw)
        all_t.extend([t_val]*nw)
        all_l.extend([l_val]*nw)
        all_r.extend([r_val]*nw)

        blocks.append({'fontsize': fs_val, 'bottom': b_val, 'top': t_val,
                       'left': l_val, 'right': r_val, 'nwords': nw,
                       'content': content, 'possible_ref': False})

    fs_normal_text = mode(all_fs[:len(all_fs)//2])
    ref_idx_start, ref_idx_end = _get_ref_index(blocks, fs_normal_text)

    all_fs_ref = []
    if ref_idx_start:
        for block in blocks[ref_idx_start:ref_idx_end]:
            all_fs_ref.extend([block['fontsize']])
            block['possible_ref'] = True

    specs = {'fontsize': all_fs, 'bottom': all_b, 'top': all_t,
             'left': all_l, 'right': all_r, 'nwords': n_words,
             'fontsize_ref': all_fs_ref}

    return blocks, specs


def get_clean_block_text(text):
    if len(text) > 0:
        try:
            blocks, specs = _get_block_text_specs(text)
            fs_normal_text = mode(
                specs['fontsize'][:len(specs['fontsize'])//2])

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
            if specs['fontsize_ref']:
                fs_ref_text = mode(
                    specs['fontsize_ref'][:len(specs['fontsize_ref'])//2])
                refblocks = [blk['content'] for blk in blocks
                             if (blk['fontsize'] == fs_ref_text and blk['possible_ref'])]
                ref_text = ' '.join(refblocks)
                ref_text = ' '.join(ref_text.splitlines())
                for mark in ref_markers:
                    ref_text = ref_text.replace(mark, '')

        except Exception as e:
            print(main_text)
            raise e

    return main_text, ref_text


def _extract_blocks_pdfminer(
        pdf_path: str,
        valid_pages: Optional[List[int]] = None) -> dict:

    sections = {"full_text": None,
                "main_text": None,
                "ref_text": None}

    if valid_pages == 0:
        return sections

    num_pages = get_pdf_num_pages(pdf_path)

    # Determine the most common font size (mode)
    page_width, page_height = 595, 842
    all_text_blocks, all_fontsize_blocks = [], []
    all_Bottom_blocks, all_Top_blocks = [], []
    all_Left_blocks, all_Right_blocks = [], []
    all_Bottom_words, all_Top_words = [], []
    all_Left_words, all_Right_words = [], []
    all_nwords_blocks = []
    reached_end = False
    if num_pages > 0:
        for p, page_layout in enumerate(extract_pages(pdf_path, page_numbers=valid_pages)):
            if reached_end:
                break
            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue

                left, bottom, right, top = element.bbox
                left /= page_width
                right = 1 - right/page_width
                bottom /= page_height
                top = 1 - top/page_height

                filtered_text, fontsizes = _filter_text_block(element)
                all_text_blocks.append(filtered_text)
                all_fontsize_blocks.append(mode(fontsizes))
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

    else:
        logger.warning(f"PDF {pdf_path} likely corrupted")

    q_th = 0.02
    nwords_th = 10

    bottom_th = np.quantile(all_Bottom_words, q_th)
    top_th = np.quantile(all_Top_words, q_th)
    left_th = np.quantile(all_Left_words, q_th)
    right_th = np.quantile(all_Right_words, q_th)

    blocks = [block for i, block in enumerate(zip(all_text_blocks, all_fontsize_blocks,
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
        f"b=={bottom: .1f}**"
        f"t=={top: .1f}**"
        f"l=={left: .1f}**"
        f"r=={right: .1f}**"
        f"\n{text}"
        for (text, size, bottom, top, left, right) in blocks)

    sections["main_text"], sections["ref_text"] = get_clean_block_text(
        sections["full_text"])

    return sections


def extract_pdf_sections(
        pdf_path: str,
        engine: str = "pdfminer",
        valid_pages: List[int] = None) -> dict:

    if "pdfminer" in engine.lower():
        return _extract_blocks_pdfminer(pdf_path, valid_pages)
    else:
        raise Exception("No other option than pdfminer implemented yet")


def get_valid_pages_PDFs(
        papers: List[PaperInfo],
        directory: str,
        timeout: int = 0.5,
        num_processes: int = 1):

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
                        valid_pages = get_valid_pages(filepath,
                                                      timeout=timeout,
                                                      num_processes=num_processes)

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


def collect_extracted_ids(directory: str):
    dic_list = glob.glob(os.path.join(directory, "extracted", "*.pkl"))
    return [Path(f).stem for f in dic_list]


def collect_extracted_PDFs(directory: str):
    dic_list = glob.glob(os.path.join(directory, "extracted", "*.pkl"))
    final_results = []
    for paper_file in dic_list:
        if os.path.isfile(paper_file):
            try:
                with open(paper_file, "rb") as f:
                    final_results.append(pickle.load(f))
                    os.remove(paper_file)
            except Exception as e:
                print(e)

    return final_results


def extract_worker_init(
        downloads_dir: str,
        keepPDF: Optional[bool] = False,
        timeout: Optional[float] = 60):
    global pdfextractor
    pdfextractor = PDFextractor(downloads_dir,
                                keepPDF=keepPDF,
                                timeout=timeout)


def extract_PDFs_workers(papers):
    global pdfextractor
    output = pdfextractor.extractPDFs(papers)
    return output


def _save_data(data, directory, filename):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    data.to_parquet(filepath, engine="pyarrow",
                    compression="snappy", index=True)


def extracted_to_DB(textdata, metadata):
    assert textdata['paperId'].equals(metadata['paperId'])
    id0 = textdata['paperId'].iloc[0]
    output_dir_DB = Path(os.path.join(config.VECTORDB_PATH, "extracted", id0))
    os.makedirs(output_dir_DB, exist_ok=False)

    fp_text = output_dir_DB / "text.extracted"
    textdata.to_parquet(fp_text.with_suffix(".inprogress"), engine="pyarrow",
                        compression="snappy", index=True)
    fp_text.with_suffix(".inprogress").rename(fp_text)

    fp_meta = output_dir_DB / "metadata.extracted"
    textdata.to_parquet(fp_meta.with_suffix(".inprogress"), engine="pyarrow",
                        compression="snappy", index=True)
    fp_meta.with_suffix(".inprogress").rename(fp_meta)


def collect_extracted_batch(directory: Path):
    if not os.path.isdir(directory):
        return None
    folder_list = os.listdir(directory)
    extracted_batches = []
    for extracted_id in folder_list:
        if os.path.isdir(os.path.join(directory, extracted_id)):
            text_path = glob.glob(os.path.join(
                directory, extracted_id, "text.extracted"))
            metadata_path = glob.glob(os.path.join(
                directory, extracted_id, "metadata.extracted"))
            if text_path and metadata_path:
                extracted_batches.append(extracted_id)

    return extracted_batches


def extract_PDFs(
        years: Optional[int] = None,
        filters: Optional[List[Any] |
                          List[List[Any]]] = None,
        n_jobs: Optional[int] = pdf_config['n_jobs_extract'],
        timeout_per_article: Optional[float] = pdf_config['timeout_per_article'],
        pdfs_dir: Optional[str | Path] = pdf_config['save_pdfs_to'],
        keep_pdfs: Optional[bool] = pdf_config['keep_pdfs'],
        push_to_vectorDB: Optional[bool] = False,
        done_event: Optional[bool] = None):

    if done_event is None:
        class DummyEvent:
            def is_set(self): return True
        done_event = DummyEvent()

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    if pdfs_dir is not None:
        downloads_dir_base = pdfs_dir
    else:
        downloads_dir_base = config.DATA_PATH / "tmp" / "download"

    # filters = [['abstract','==','None']]
    n_jobs = min(n_jobs, 2 * multiprocessing.cpu_count() // 3)
    timeout_per_worker = timeout_per_article * n_jobs
    timeout_new_download = 60

    for year in years:
        downloads_dir = downloads_dir_base / str(year)

        logger.info(f"Extracting pdfs for year {year}")
        paperdata = load_data(subsets=year,
                              data_types=["metadata", "text"])
        if filters is not None:
            paperdata_filt = load_data(subsets=year,
                                       data_types=[
                                           "text", "metadata"],
                                       filters=filters)
        else:
            paperdata_filt = paperdata

        paperInfo = pd.merge(paperdata_filt["text"][["paperId", "openAccessPdf"]],
                             paperdata_filt["metadata"][[
                                 "paperId", "authorName"]],
                             on="paperId",
                             how="inner")

        processed_ids, new_ids = [], []
        ids_to_save = []
        year_ids = set(paperInfo["paperId"])
        endtime = time.time() + timeout_per_article * len(paperInfo)
        while new_ids or not done_event.is_set():
            downloaded_ids = collect_downloaded_ids(downloads_dir)
            paper_ids = [id for id in downloaded_ids if id in year_ids]

            new_ids = [id for id in paper_ids if id not in processed_ids]

            if not new_ids and not done_event.is_set():
                time.sleep(5)
                logger.info(
                    f"no new data. Will wait {endtime - time.time():.1f} more seconds")
                continue

            processed_ids = downloaded_ids

            paperInfo_new = paperInfo[paperInfo["paperId"].isin(new_ids)]
            minibatch_size = 1  # 100
            # len(paperInfo_s) // 10 #2 * n_jobs * minibatch_size
            batch_size = n_jobs
            Npapers = len(paperInfo_new)
            tqdm = get_tqdm()
            with tqdm(range(math.ceil(Npapers / (batch_size + 1))),
                      desc=f"Extracting {Npapers} new papers") as pbar:
                try:
                    for i in pbar:
                        TimoutRaised = False
                        batch = paperInfo_new.iloc[i *
                                                   batch_size: (i+1)*batch_size].to_dict(orient="records")

                        Nbatch = len(batch)
                        minibatches = [batch[k * minibatch_size: (k + 1) * minibatch_size]
                                       for k in range(Nbatch // minibatch_size + 1)]

                        minibatches = [b for b in minibatches
                                       if len(b) > 0 and b[0] is not None]

                        final_results = []
                        Nminibatches = len(minibatches)
                        if minibatches:
                            extract_pool = multiprocessing.Pool(
                                processes=n_jobs,
                                initializer=partial(
                                    extract_worker_init,
                                    downloads_dir=downloads_dir,
                                    keepPDF=keep_pdfs,
                                    timeout=60))

                            with extract_pool:
                                async_extract = [extract_pool.apply_async(extract_PDFs_workers, (b,))
                                                 for b in minibatches]

                                pending_extract = set(async_extract)

                                end_time = time.time() + timeout_per_article * batch_size
                                old_pending_articles = batch_size
                                while pending_extract:
                                    if time.time() > end_time:
                                        logger.info(
                                            f"{len(pending_extract)} extractions out of {Nminibatches} timed out. Terminating the pool.")

                                        TimoutRaised = True
                                        extract_pool.terminate()
                                        extract_pool.join()
                                        break

                                    for task in list(pending_extract):
                                        if task.ready():
                                            pending_extract.remove(task)

                                    pbar.set_postfix(
                                        {"extraction tasks": f"{len(pending_extract)} left"})

                                    time.sleep(1)

                                    extracted_ids = collect_extracted_ids(
                                        directory=downloads_dir)
                                    pending_ids = [
                                        id for id in new_ids if id not in extracted_ids]
                                    n_pending_articles = len(pending_ids)

                                    if n_pending_articles != old_pending_articles:
                                        if pending_extract:
                                            extratime = (
                                                timeout_per_worker / 2 * n_pending_articles / len(pending_extract))
                                        else:
                                            extratime = 0

                                        pbar.set_postfix({"extraction tasks": f"{len(pending_extract)} still running, \
                                                        {extratime} seconds left"})

                                        end_time = time.time() + extratime
                                        old_pending_articles = len(pending_ids)
                                    else:
                                        end_time = time.time() + timeout_per_worker

                            if TimoutRaised:
                                # CHECK THAT THE FOLLOWING WORKS!
                                # Checking valid pages to avoid pdfminer getting stuck
                                # This will change results in place
                                extracted_ids = collect_extracted_ids(
                                    directory=downloads_dir)
                                pending_ids = [
                                    id for id in new_ids if id not in extracted_ids]
                                pending_papers = [
                                    b for b in batch if b["paperId"] in pending_ids]

                                logger.info(f"checking valid pages for {len(pending_papers)}" +
                                            "articles that timed out...")

                                get_valid_pages_PDFs(papers=pending_papers,
                                                     directory=downloads_dir,
                                                     timeout=1)

                                n_pending = len(pending_papers)
                                minib_size = math.ceil(
                                    len(pending_papers) / (n_jobs + 1))

                                pending_minibatches = [pending_papers[k * minib_size: (k + 1) * minib_size]
                                                       for k in range(n_jobs + 1)]

                                pending_minibatches = [b for b in pending_minibatches
                                                       if len(b) > 0 and b[0] is not None]

                                n_pending = len(pending_minibatches)
                                pbar.set_postfix(
                                    {"status": f"will process now {(n_pending)} articles..."})

                                with multiprocessing.Pool(processes=n_jobs) as pool:
                                    async_extract = [pool.apply_async(extract_PDFs_workers, (b,))
                                                     for b in pending_minibatches]

                                    pending = set(async_extract)
                                    end_time_extra = time.time() + timeout_per_article * n_pending
                                    while pending:
                                        if time.time() > end_time_extra:
                                            logger.info(f"{len(pending)} tasks out of {n_pending}" +
                                                        "timed out after checking valid pdf pages.")

                                            TimoutRaised = True
                                            pool.terminate()
                                            pool.join()
                                            break

                                        for task in list(pending):
                                            if task.ready():
                                                pending.remove(task)
                                        if pending:
                                            time.sleep(0.5)

                            pbar.set_postfix(
                                {"status": "gathering results..."})
                            final_results = collect_extracted_PDFs(
                                downloads_dir)

                            if final_results:
                                pdfdata = pd.DataFrame(
                                    [pdf for pdf in final_results])

                                ids_to_save.extend(pdfdata["paperId"])

                                pbar.set_postfix({"status": f"{len(pdfdata)} " +
                                                  f"articles extracted out of {Nminibatches}"})

                                paperdata["text"] = (pdfdata.set_index("paperId").combine_first(
                                    paperdata["text"].set_index("paperId")).reset_index(drop=False))

                                paperdata["text"] = paperdata["text"].fillna(
                                    'None')
                                paperdata["text"]["authorName"] = paperdata["metadata"]["authorName"]

                            if len(ids_to_save) > 256:
                                pbar.set_postfix(
                                    {"status": f"Saving {len(ids_to_save)} new articles"})
                                output_dir_year = os.path.join(
                                    config.TEXTDATA_PATH, str(year))
                                _save_data(paperdata["text"],
                                           directory=output_dir_year,
                                           filename=f"text_{year}.parquet")

                                # CHECK IF THE FOLLOWING WORKS
                                if push_to_vectorDB:
                                    DBmask = paperdata['text']["paperId"].isin(
                                        ids_to_save)
                                    text_to_DB = paperdata['text'].loc[DBmask]
                                    metadata_to_DB = paperdata['metadata'].loc[DBmask]
                                    extracted_to_DB(text_to_DB, metadata_to_DB)

                                ids_to_save = []
                except Exception as e:
                    pbar.close()
                    raise e

            endtime = time.time() + timeout_new_download

        if len(ids_to_save) > 0:
            output_dir_year = os.path.join(config.TEXTDATA_PATH, str(year))
            _save_data(paperdata["text"],
                       directory=output_dir_year,
                       filename=f"text_{year}.parquet")

            if push_to_vectorDB:
                DBmask = paperdata['text']["paperId"].isin(ids_to_save)
                text_to_DB = paperdata['text'].loc[DBmask]
                metadata_to_DB = paperdata['metadata'].loc[DBmask]
                extracted_to_DB(text_to_DB, metadata_to_DB)

        if not keep_pdfs:
            shutil.rmtree(downloads_dir)

    return paperdata["text"]


# Previous approach to extract sections. Should rather be done prior to push to vectorDB

# def pdf_main_fontsize(
#         pdf_path: str,
#         valid_pages: Optional[List[int]] = None) -> Optional[float]:
#     """Estimate the main font size used in the document."""

#     try:
#         sizes = [
#             character.size
#             for page_layout in extract_pages(pdf_path, page_numbers=valid_pages)
#             for element in page_layout
#             if isinstance(element, LTTextContainer)
#             for text_line in element
#             if isinstance(text_line, LTTextLine)
#             for character in text_line
#             if isinstance(character, LTChar)]
#         return mode(sizes) if sizes else None
#     except Exception as e:
#         logger.warning(f"Font size extraction failed for {pdf_path}: {e}")
#         return None

# def _get_markers():
#     last_section_markers = {
#         "acknowledgment",
#         "acknowledgement",
#         "acknowlegment",
#         "reference",
#     }
#     month_markers = {
#         "january",
#         "february",
#         "march",
#         "april",
#         "may",
#         "june",
#         "july",
#         "august",
#         "september",
#         "november",
#         "december",
#         "jan",
#         "feb",
#         "mar",
#         "apr",
#         "may",
#         "jun",
#         "jul",
#         "aug",
#         "sep",
#         "nov",
#         "dec",
#     }
#     year_markers = {str(y) for y in range(1900, 2030)}
#     section_list = {
#         "introduction",
#         "results",
#         "discussion",
#         "conclusions",
#         "methods",
#         "materials",
#         "experimental",
#         "materials and methods",
#         "experimental procedure",
#         "related work",
#         "i.",
#         "ii.",
#         "iii.",
#         "iv.",
#         "v.",
#         "vi.",
#     }
#     return last_section_markers, month_markers, year_markers, section_list

# def _extract_sections_pdfminer(
#         pdf_path: str,
#         authorlist: str = None,
#         valid_pages: Optional[List[int]] = None,
#         possible_section_headings: Optional[Set] = None,
#         size_threshold: Optional[float] = None) -> dict:
#     author_last_names = (
#         [name.split()[-1] for name in authorlist.split(",") if name.split()]
#         if authorlist
#         else None)

#     last_section_names, month_markers, year_markers, section_list = _get_markers()

#     section_list = possible_section_headings or section_list

#     size_threshold = size_threshold if size_threshold is not None else 0.9

#     sections = {
#         "pdf_abstract": None,
#         "pdf_introduction": None,
#         "pdf_results": None,
#         "pdf_discussion": None,
#         "pdf_methods": None,
#         "full_text": None}

#     if valid_pages == 0:
#         return sections

#     # Determine the most common font size (mode)
#     size_mode = pdf_main_fontsize(pdf_path, valid_pages)

#     page_width, page_height = 595, 842
#     nword_abstract_th, nword_sections_th = 30, 30
#     text_blocks, tag_blocks, nwords_in_blocks = [], [], []
#     all_text_blocks = []
#     all_size_blocks = []
#     section_to_find, section_heading, tag = "AUTHORS", "UNDEFINED", "UNDEFINED"
#     reached_end = False
#     if size_mode is not None:
#         for p, page_layout in enumerate(extract_pages(pdf_path, page_numbers=valid_pages)):
#             if reached_end:
#                 break
#             for element in page_layout:
#                 if not isinstance(element, LTTextBox):
#                     continue

#                 x0, y0, x1, y1 = element.bbox
#                 if not (y0 > 0.05 * page_height
#                         and y1 < 0.95 * page_height
#                         and x0 > 0.05 * page_width
#                         and x1 < 0.95 * page_width):
#                     continue

#                 filtered_text, sizes = _filter_text_block(element)

#                 word_list = re.split(r"[\n\s]+", filtered_text.lower().strip())
#                 nwords = len(word_list)

#                 if any(end_section in " ".join(word_list[:3])
#                        for end_section in last_section_names):
#                     reached_end = True
#                     continue

#                 if not reached_end:
#                     all_text_blocks.append(filtered_text)
#                     all_size_blocks.append(mode(sizes))

#                 # removing everything before the author block as well as the correspondance fields
#                 if p <= 1 and author_last_names:
#                     nauthors_detected = sum(
#                         lastname.lower() in " ".join(word_list)
#                         for lastname in author_last_names)
#                     if (nauthors_detected >= 0.5 * len(author_last_names)
#                         and y0 > 0.3 * page_height
#                             and section_to_find == "AUTHORS"):
#                         text_blocks, tag_blocks, nwords_in_blocks = [], [], []
#                         section_to_find = "ABSTRACT"
#                         filtered_text = []
#                     if nauthors_detected > 0 and ("@" in filtered_text or "correspond" in filtered_text):
#                         filtered_text = []

#                 # removing blocks likely headers with publication date
#                 if (any([m in word_list for m in month_markers])
#                     and any([y in word_list for y in year_markers])
#                         and nwords < 10):
#                     continue

#                 # removing figure captions
#                 if any([figname in word_list[0]
#                         for figname in ["fig", "figure", "table", "image"]]):
#                     continue

#                 # removing previous block if likely a header but not followed by capitalized paragraph
#                 if (filtered_text
#                     and not filtered_text[0].isupper()
#                     and nwords_in_blocks
#                         and nwords_in_blocks[-1] <= 3):
#                     text_blocks.pop()
#                     tag_blocks.pop()
#                     nwords_in_blocks.pop()
#                 elif (filtered_text
#                       and filtered_text.strip()
#                       and filtered_text.strip()[0].isupper()
#                       and nwords_in_blocks
#                       and nwords_in_blocks[-1] <= 3
#                       and text_blocks
#                       and any(h in re.sub(r"[\n\s]+", " ", text_blocks[-1].lower())
#                               for h in section_list)):
#                     section_heading = "".join([w.upper()
#                                                for w in re.sub(r"[\d.]", "", text_blocks[-1])])
#                     if nwords > nword_sections_th:
#                         tag_blocks[-1] = section_heading

#                 if (not reached_end
#                     and filtered_text
#                     and (max(sizes, default=0) >= size_threshold * size_mode
#                          or nwords > 50)):
#                     if section_to_find == "ABSTRACT" and nwords > nword_abstract_th:
#                         tag = "ABSTRACT"
#                         if word_list[-1][-1] == ".":
#                             section_to_find = "INTRODUCTION"
#                     elif section_to_find == "INTRODUCTION":
#                         if nwords > nword_sections_th:
#                             tag = "INTRODUCTION"
#                             section_heading, section_to_find = (
#                                 "INTRODUCTION", "NEXTHEADING")
#                         else:
#                             tag = "UNDEFINED"
#                     elif (section_to_find == "NEXTHEADING" and nwords > nword_sections_th):
#                         tag = section_heading

#                     text_blocks.append(filtered_text)
#                     tag_blocks.append(tag)
#                     nwords_in_blocks.append(nwords)
#     else:
#         logger.warning(f"PDF {pdf_path} likely corrupted")

#     sections["pdf_abstract"] = "\n".join(
#         text
#         for t, text in zip(tag_blocks, text_blocks)
#         if any(s in t.lower() for s in {"abstract"}))

#     sections["pdf_introduction"] = "\n".join(
#         text
#         for t, text in zip(tag_blocks, text_blocks)
#         if any(s in t.lower() for s in {"introduction", "related work", "i.", "ii."}))

#     sections["pdf_results"] = "\n".join(
#         text
#         for t, text in zip(tag_blocks, text_blocks)
#         if any(s in t.lower() for s in {"results", "experiment", "i.", "ii."}))

#     sections["pdf_discussion"] = "\n".join(
#         text
#         for t, text in zip(tag_blocks, text_blocks)
#         if any(s in t.lower() for s in {"discussion", "conclusion", "v.", "vi."}))

#     sections["pdf_methods"] = "\n".join(
#         text
#         for t, text in zip(tag_blocks, text_blocks)
#         if any(
#             s in t.lower()
#             for s in {
#                 "methods",
#                 "materials",
#                 "experimental",
#                 "materials and methods",
#                 "experimental procedure"}))

#     sections["full_text"] = "\n**BLOCK**".join(f"{size}**\n{text}"
#                                                 for text, size in zip(all_text_blocks, all_size_blocks))

#     return sections
