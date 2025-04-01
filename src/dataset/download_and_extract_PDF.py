import re
from pathlib import Path
from multiprocessing import Process, Event
import asyncio
import pandas as pd
from datetime import datetime
from typing import Optional, List, Any

from dataIO.load import load_data
from pdf.extract_PDF import extract_PDFs
from pdf.download_PDF import download_PDFs

import logging
from src.logger import get_logger
from src.utils import load_params

logger = get_logger(__name__, level=logging.INFO)

params = load_params()
pdf_config = params["pdf"]


def download_and_extract(
        years: Optional[int] = None,
        filters: Optional[List[Any] | List[List[Any]]] = None,
        n_jobs: Optional[List[int]] = [
            pdf_config['n_jobs_download'], pdf_config['n_jobs_extract']],
        timeout_loadpage: Optional[float] = pdf_config['timeout_loadpage'],
        timeout_startdw: Optional[float] = pdf_config['timeout_startdw'],
        save_pdfs_to: Optional[str | Path] = pdf_config['save_pdfs_to'],
        timeout_per_article: Optional[float] = pdf_config['timeout_per_article'],
        keep_pdfs: Optional[bool] = pdf_config['keep_pdfs'],
        push_to_vectorDB: Optional[bool] = False):

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    done_event = Event()

    download_proc = Process(
        target=download_PDFs,
        args=(years, filters, n_jobs[0],
              timeout_loadpage, timeout_startdw,
              save_pdfs_to,
              done_event))

    extract_proc = Process(
        target=extract_PDFs,
        args=(years, filters, n_jobs[1],
              timeout_per_article, save_pdfs_to,
              keep_pdfs, push_to_vectorDB,
              done_event))

    extract_proc.start()
    download_proc.start()

    download_proc.join()
    extract_proc.join()

# import os
# import glob
# import shutil
# import time
# from datetime import datetime
# from functools import partial

# from typing import List, Optional, Any, TypedDict
# import pickle
# import pandas as pd
# import math

# from .utils_PDF import extract_pdf_sections, get_valid_pages
# from .utils_Download import PDFdownloader, close_all_chrome_sessions
# from .dataset_loader import load_data

# from tqdm import tqdm
# import multiprocessing
# import atexit

# import config
# import logging
# from src.utils import load_params
# from src.logger import get_logger

# logger = get_logger(__name__, level=logging.INFO)

# params = load_params()
# pdf_config = params["pdf"]


# class PaperInfo(TypedDict):
#     paperId: str
#     openAccessPdf: str


# class PDFextractor:
#     def __init__(
#             self,
#             directory: str,
#             keepPDF: Optional[bool] = False,
#             timeout: Optional[int] = 60):

#         self.directory = directory
#         self.keepPDF = keepPDF
#         self.timeout = timeout
#         self.processed_dir = os.path.join(self.directory, "extracted")
#         os.makedirs(self.processed_dir, exist_ok=True)

#     def extractPDFs(self, papers: List[PaperInfo]):
#         # dir_path_orig = [os.path.join(downloads_dir, pdf_data['paperId']) for pdf_data in textdata]
#         dir_list = [{"id": pdf_data["paperId"], "time": None}
#                     for pdf_data in papers]
#         while dir_list:
#             # if os.path.isdir(downloads_dir):
#             #     dir_list = os.listdir(downloads_dir) if dir_list is None else dir_list
#             # dir_list.sort(key=lambda name: os.path.getmtime(os.path.join(downloads_dir, name)))
#             for dirname in dir_list:
#                 dir_path = os.path.join(self.directory, dirname["id"])
#                 if os.path.isdir(dir_path):
#                     if not dirname["time"]:
#                         dirname["time"] = os.path.getmtime(dir_path)
#                     if glob.glob(os.path.join(dir_path, "*.pdf")):
#                         filepath = glob.glob(
#                             os.path.join(dir_path, "*.pdf"))[0]
#                         paper_data = next(
#                             (pdf_data for pdf_data in papers if pdf_data["paperId"] == dirname["id"]), None)

#                         if paper_data:
#                             if "valid_pages" not in paper_data.keys():
#                                 paper_data["valid_pages"] = None
#                             pdf_data = extract_pdf_sections(filepath,
#                                                             valid_pages=paper_data["valid_pages"])

#                             for key in pdf_data.keys():
#                                 paper_data[key] = pdf_data[key]
#                             paper_data["pdf_status"] = "extracted"

#                             dic_path = os.path.join(self.processed_dir,
#                                                     dirname["id"] + ".pkl")

#                             with open(dic_path, "wb") as f:
#                                 pickle.dump(paper_data, f,
#                                             protocol=pickle.HIGHEST_PROTOCOL)

#                         dir_list.remove(dirname)
#                         if not self.keepPDF:
#                             shutil.rmtree(dir_path)
#                     elif time.time() - dirname["time"] > self.timeout:
#                         crfilepath = glob.glob(
#                             os.path.join(dir_path, "*.crdownload"))
#                         if not crfilepath or \
#                                 time.time() - os.path.getmtime(crfilepath[0]) > self.timeout:
#                             dir_list.remove(dirname)
#                             if not self.keepPDF:
#                                 shutil.rmtree(dir_path)
#         return papers


# def get_valid_pages_PDFs(
#         papers: List[PaperInfo],
#         directory: str,
#         timeout: int = 0.1,
#         num_processes: int = 1):

#     # Extracting pdfs from downloads that took too long to be processed immediately
#     # downloads_dir = "/tmp/s2tmp"
#     if os.path.isdir(directory):
#         dir_list = os.listdir(directory)
#         dir_list.sort(key=lambda name: os.path.getmtime(
#             os.path.join(directory, name)))
#         for dirname in dir_list:
#             dir_path = os.path.join(directory, dirname)
#             if os.path.isdir(dir_path):
#                 if glob.glob(os.path.join(dir_path, "*.pdf")):
#                     filepath = glob.glob(os.path.join(dir_path, "*.pdf"))[0]
#                     paper_data = next((pdf_data
#                                        for pdf_data in papers
#                                        if pdf_data["paperId"] == dirname), None)

#                     if paper_data:
#                         valid_pages = get_valid_pages(filepath,
#                                                       timeout=timeout,
#                                                       num_processes=num_processes)

#                         paper_data["valid_pages"] = valid_pages
#     return papers


# def get_unprocessed_PDFs(
#         papers: List[PaperInfo],
#         directory: str):

#     filtered_papers = []
#     if os.path.isdir(directory):
#         filtered_papers = []
#         for paper in papers:
#             try:
#                 pdf_path = os.path.join(directory, paper["paperId"])
#                 dic_path = os.path.join(directory,
#                                         "extracted",
#                                         paper["paperId"] + ".pkl")

#                 if os.path.isdir(pdf_path) and not os.path.isfile(dic_path):
#                     if glob.glob(os.path.join(pdf_path, "*.pdf")):
#                         filtered_papers.append(paper)
#             except Exception:
#                 pass

#     return filtered_papers


# def collect_extracted_PDFs(directory: str):
#     dic_list = glob.glob(os.path.join(directory, "extracted", "*.pkl"))
#     final_results = []
#     for paper_file in dic_list:
#         if os.path.isfile(paper_file):
#             try:
#                 with open(paper_file, "rb") as f:
#                     final_results.append(pickle.load(f))
#                     os.remove(paper_file)
#             except Exception as e:
#                 print(e)

#     return final_results


# def download_worker_init(
#         downloads_dir: str,
#         source: str = "s2",
#         timeout_loadpage: Optional[float] = 15,
#         timeout_startdw: Optional[float] = 5):
#     global downloader
#     downloader = PDFdownloader(downloads_dir,
#                                source=source,
#                                headless=False,
#                                worker_id=os.getpid(),
#                                timeout_loadpage=timeout_loadpage,
#                                timeout_startdw=timeout_startdw)

#     atexit.register(download_worker_close)


# def download_worker_close():
#     global downloader
#     downloader.close()


# def extract_worker_init(
#         downloads_dir: str,
#         keepPDF: Optional[bool] = False,
#         timeout: Optional[float] = 60):
#     global pdfextractor
#     pdfextractor = PDFextractor(downloads_dir,
#                                 keepPDF=keepPDF,
#                                 timeout=timeout)


# def download_PDFs(papers):
#     global downloader
#     output = [downloader.download(paper) for paper in papers]
#     while not downloader.is_download_finished():
#         time.sleep(1)
#     downloader.reset_queue()
#     return output


# def extract_PDFs(papers):
#     global pdfextractor
#     output = pdfextractor.extractPDFs(papers)
#     return output


# def download_and_extract_PDFs(
#         years: Optional[int] = None,
#         filters: Optional[List[Any] |
#                           List[List[Any]]] = None,
#         n_jobs: Optional[int] = pdf_config['n_jobs_extract'],
#         timeout_per_article: Optional[float] = pdf_config['timeout_per_article'],
#         timeout_loadpage: Optional[float] = pdf_config['timeout_loadpage'],
#         timeout_startdw: Optional[float] = pdf_config['timeout_startdw'],
#         local_source: Optional[str] = pdf_config['save_pdfs_to']):

#     years = [datetime.now().year] if years is None else years
#     years = [years] if not isinstance(years, list) else years

#     if local_source:
#         downloads_dir = local_source
#     else:
#         downloads_dir = config.DATA_PATH / "tmp"
#     os.makedirs(downloads_dir, exist_ok=True)

#     # filters = [['abstract','==','None']]
#     if not isinstance(n_jobs, list):
#         n_jobs = min(n_jobs, 2 * multiprocessing.cpu_count() // 3)
#         n_jobs_dwload = min(20, n_jobs // 2)
#         n_jobs_extract = n_jobs - n_jobs_dwload
#     else:
#         n_jobs_dwload, n_jobs_extract = n_jobs

#     n_jobs_total = n_jobs_dwload + n_jobs_extract
#     timeout_per_worker = timeout_per_article * n_jobs_total

#     for year in years:
#         logger.info(f"Downloading pdfs for year {year}")
#         paperdata = load_data(subsets=year,
#                                  data_types=["metadata", "text"])

#         if filters is not None:
#             paperdata_filt = load_data(subsets=year,
#                                           data_types=[
#                                               "text", "metadata"],
#                                           filters=filters)
#         else:
#             paperdata_filt = paperdata

#         paperInfo = pd.merge(paperdata_filt["text"][["paperId", "openAccessPdf"]],
#                              paperdata_filt["metadata"][[
#                                  "paperId", "authorName"]],
#                              on="paperId",
#                              how="inner")

#         paperInfo_s = {"http": paperInfo[paperInfo["openAccessPdf"].str.startswith("http")],
#                        "gs": paperInfo[paperInfo["openAccessPdf"].str.startswith("gs:")], }
#         external_sources = {"http", "gs"} if not local_source else {None}
#         for source in external_sources:
#             if source is not None:
#                 paperInfo_s = paperInfo[paperInfo["openAccessPdf"].str.startswith(
#                     source)]
#             else:
#                 paperInfo_s = paperInfo

#             if not paperInfo_s.empty:
#                 minibatch_size = 20  # 100
#                 # len(paperInfo_s) // 10 #2 * n_jobs * minibatch_size
#                 batch_size = 3000
#                 Npapers = 3000  # len(paperInfo_s)
#                 with tqdm(range(math.ceil(Npapers / (batch_size + 1))),
#                           desc=f"Year {year} / {len(years)} years ({Npapers} papers)",
#                           unit="batch") as pbar:
#                     try:
#                         for i in pbar:
#                             if not local_source:
#                                 shutil.rmtree(downloads_dir)
#                                 os.makedirs(downloads_dir, exist_ok=True)
#                             close_all_chrome_sessions()

#                             print(downloads_dir)

#                             TimoutRaised = False
#                             batch = paperInfo_s.iloc[i *
#                                                      batch_size: (i+1)*batch_size].to_dict(orient="records")

#                             Nbatch = len(batch)
#                             minibatches = [batch[k * minibatch_size: min(Nbatch, (k + 1) * minibatch_size)]
#                                            for k in range(Nbatch // minibatch_size + 1)]

#                             minibatches = [b for b in minibatches
#                                            if len(b) > 0 and b[0] is not None]

#                             final_results, dwload_results = [], []
#                             Nminibatches = len(minibatches)
#                             if minibatches:
#                                 extract_pool = multiprocessing.Pool(
#                                     processes=n_jobs_extract,
#                                     initializer=partial(
#                                         extract_worker_init,
#                                         downloads_dir=downloads_dir,
#                                         keepPDF=True if local_source else False,
#                                         timeout=60))

#                                 if not local_source:
#                                     download_pool = multiprocessing.Pool(
#                                         processes=n_jobs_dwload,
#                                         initializer=partial(
#                                             download_worker_init,
#                                             downloads_dir=downloads_dir,
#                                             timeout_loadpage=timeout_loadpage,
#                                             timeout_startdw=timeout_startdw))

#                                 else:
#                                     download_pool = None

#                                 with download_pool, extract_pool:
#                                     if download_pool:
#                                         async_dwload = [download_pool.apply_async(download_PDFs, (b,))
#                                                         for b in minibatches]
#                                         pending_dwload = set(async_dwload)
#                                     else:
#                                         pending_dwload = []

#                                     async_extract = [extract_pool.apply_async(extract_PDFs, (b,))
#                                                      for b in minibatches]

#                                     pending_extract = set(async_extract)

#                                     end_time = time.time() + timeout_per_article * batch_size
#                                     old_pending_articles = batch_size
#                                     while pending_extract or pending_dwload:
#                                         # Adjust timeout wall to one many tasks of extraction are still pending still download is done
#                                         if time.time() > end_time:
#                                             logger.info(
#                                                 f"{len(pending_extract)} extractions and {len(pending_dwload)} downloads out of {Nminibatches} timed out. Terminating the pool.")

#                                             TimoutRaised = True
#                                             extract_pool.terminate()
#                                             extract_pool.join()
#                                             if download_pool:
#                                                 download_pool.terminate()
#                                                 download_pool.join()
#                                             break

#                                         for task in list(pending_extract):
#                                             if task.ready():
#                                                 pending_extract.remove(task)

#                                         for task in list(pending_dwload):
#                                             if task.ready():
#                                                 res_dw = task.get()
#                                                 dwload_results.append(res_dw)
#                                                 pending_dwload.remove(task)

#                                         if pending_extract or pending_dwload:
#                                             if pending_dwload:
#                                                 pbar.set_postfix({"download tasks": f"{len(pending_dwload)} left",
#                                                                   "extraction tasks": f"{len(pending_extract)} left"})

#                                             time.sleep(1)

#                                         if not pending_dwload:
#                                             time.sleep(timeout_per_worker)

#                                             n_pending_articles = len(get_unprocessed_PDFs(papers=batch,
#                                                                                           directory=downloads_dir))

#                                             if n_pending_articles != old_pending_articles:
#                                                 extratime = (
#                                                     timeout_per_worker / 2 * n_pending_articles / len(pending_extract))

#                                                 pbar.set_postfix({"extraction tasks": f"{len(pending_extract)} still running, \
#                                                                 {extratime} seconds left"})

#                                                 end_time = time.time() + extratime
#                                                 old_pending_articles = n_pending_articles
#                                             else:
#                                                 end_time = time.time()

#                                 if TimoutRaised:
#                                     # Checking valid pages to avoid pdfminer getting stuck
#                                     # This will change results in place
#                                     pending_articles = get_unprocessed_PDFs(papers=batch,
#                                                                             directory=downloads_dir)

#                                     pbar.set_postfix({"status": f"checking valid pages for {len(pending_articles)}" +
#                                                       "articles that timed out..."})

#                                     get_valid_pages_PDFs(papers=pending_articles,
#                                                          directory=downloads_dir,
#                                                          timeout=1)

#                                     n_pending = len(pending_articles)
#                                     minib_size = math.ceil(
#                                         len(pending_articles) / (n_jobs_total + 1))

#                                     pending_minibatches = [pending_articles[k * minib_size: (k + 1) * minib_size]
#                                                            for k in range(n_jobs_total + 1)]

#                                     pending_minibatches = [b for b in pending_minibatches
#                                                            if len(b) > 0 and b[0] is not None]

#                                     n_pending = len(pending_minibatches)
#                                     pbar.set_postfix(
#                                         {"status": f"will process now {(n_pending)} articles..."})

#                                     with multiprocessing.Pool(processes=n_jobs_total) as pool:
#                                         async_extract = [pool.apply_async(extract_PDFs, (b,))
#                                                          for b in pending_minibatches]

#                                         pending = set(async_extract)
#                                         end_time = time.time() + timeout_per_article * n_pending
#                                         while pending:
#                                             if time.time() > end_time:
#                                                 logger.info(f"{len(pending)} tasks out of {n_pending}" +
#                                                             "timed out after checking valid pdf pages.")

#                                                 TimoutRaised = True
#                                                 pool.terminate()
#                                                 pool.join()
#                                                 break

#                                             for task in list(pending):
#                                                 if task.ready():
#                                                     pending.remove(task)
#                                             if pending:
#                                                 time.sleep(0.5)

#                                 pbar.set_postfix(
#                                     {"status": "gathering results..."})
#                                 final_results = collect_extracted_PDFs(
#                                     downloads_dir)

#                                 if final_results:
#                                     print("final_results collected")
#                                     pdfdata = pd.DataFrame(
#                                         [pdf for pdf in final_results])

#                                     pbar.set_postfix({"status": f"{len(pdfdata)} " +
#                                                       f"articles extracted out of {batch_size}"})

#                                     if dwload_results:
#                                         print("collecting download results")
#                                         dwdata = pd.DataFrame([{k: d.get(k, None)
#                                                                 for k in ["paperId", "pdf_status"]}
#                                                                for r in dwload_results
#                                                                if r is not None
#                                                                for d in r
#                                                                if d is not None])
#                                         pdfdata = (pdfdata.set_index("paperId").combine_first(
#                                             dwdata.set_index("paperId")).reset_index(drop=False))
#                                         # pdfdata = (dwdata.set_index("paperId").combine_first(
#                                         #     pdfdata.set_index("paperId")).reset_index(drop=False))

#                                     paperdata["text"] = (pdfdata.set_index("paperId").combine_first(
#                                         paperdata["text"].set_index("paperId")).reset_index(drop=False))

#                                     # mask = paperdata["text"]["abstract"] == "None"
#                                     # paperdata["text"].loc[mask,
#                                     #                       "abstract"] = paperdata["text"].loc[mask, "pdf_abstract"]
#                                     paperdata["text"] = paperdata["text"].fillna(
#                                         'None')
#                                     paperdata["text"]["authorName"] = paperdata["metadata"]["authorName"]

#                         # Cleaning up download directory
#                         if not local_source:
#                             shutil.rmtree(downloads_dir)
#                             os.makedirs(downloads_dir, exist_ok=True)
#                         close_all_chrome_sessions()

#                     except Exception as e:
#                         pbar.close()
#                         raise e

#         output_dir_year = os.path.join(config.TEXTDATA_PATH, str(year))
#         os.makedirs(output_dir_year, exist_ok=True)
#         filepath = os.path.join(output_dir_year,
#                                 f"text_{year}.parquet")

#         paperdata["text"].to_parquet(filepath, engine="pyarrow",
#                                      compression="snappy", index=True)

#     return paperdata["text"]
