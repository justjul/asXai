import config
from src.utils import load_params
from .dataset_loader import load_dataset
from functools import partial
import multiprocessing
from typing import List, Optional, Any, TypedDict
import pandas as pd
import math
import datetime
import os
import glob
import shutil
import tempfile
import re
import time
from typing import Optional, TypedDict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import subprocess

import atexit

import logging
from src.logger import get_logger
from src.utils import get_tqdm
import psutil

logger = get_logger(__name__, level=logging.INFO)

params = load_params()
pdf_config = params["pdf"]


class PaperInfo(TypedDict):
    paperId: str
    openAccessPdf: str


class PDFdownloader:
    def __init__(self,
                 downloads_dir: str,
                 source: str = "s2",
                 headless: Optional[bool] = False,
                 worker_id: Optional[int] = 0,
                 timeout_loadpage: Optional[float] = 15,
                 timeout_startdw: Optional[float] = 5):
        self.downloads_dir = downloads_dir
        self.source = source
        self.queue = []
        self.headless = headless
        self.worker_id = worker_id

        temp_dir = tempfile.gettempdir()
        self.user_data_dir = tempfile.mkdtemp(
            prefix=f"chrome-profile-{self.worker_id}-", dir=temp_dir)
        if self.source == "s2":
            self.driver = self._s2_init_driver(headless)
        else:
            self.driver = None

        self.timeout_startdw = timeout_startdw
        self.timeout_loadpage = timeout_loadpage
        self.block_markers = {
            "captcha",
            "verify you are human",
            "not a robot",
            "forbidden",
            "error: 500",
            "this site can’t be reached"}

        self.error_markers = {
            "just a moment",
            "privacy error",
            "validate user",
            "error 500"}

    def _s2_init_driver(self, headless: bool = False):
        chromedriver_path = shutil.which("chromedriver")
        if not chromedriver_path:
            raise FileNotFoundError(
                "chromedriver not found in your PATH. Please install it or add it to PATH.")
        service = Service(executable_path=chromedriver_path)

        options = webdriver.ChromeOptions()
        options.add_argument(f"--user-data-dir={str(self.user_data_dir)}")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        if headless:
            options.add_argument("--headless=new")
        prefs = {
            "plugins.always_open_pdf_externally": True,
            "download.prompt_for_download": False,
            "download.default_directory": str(self.downloads_dir),
            "download.directory_upgrade": True}
        options.add_experimental_option("prefs", prefs)
        chromedriver_path = shutil.which("chromedriver")

        waitforSession = 0
        max_retries = 3
        retry_delay = 3
        while waitforSession < max_retries:
            try:
                driver = webdriver.Chrome(service=service, options=options)
                break
            except Exception as e:
                logger.warning(
                    f"Session creation failed.\n\
                        {e}\n\
                        Retrying ({waitforSession + 1}/{max_retries})..."
                )
                time.sleep(retry_delay)  # Wait before retrying
                waitforSession += 1
        else:
            logger.error(
                f"Failed to create a Selenium session after {max_retries} attempts"
            )
            raise RuntimeError(
                f"Failed to create a Selenium session after {max_retries} attempts"
            )

        time.sleep(0.5)

        return driver

    def close(self):
        if self.driver is not None:
            self.driver.quit()
            shutil.rmtree(self.user_data_dir, ignore_errors=True)

    def reset_queue(self):
        self.queue = []

    def download(self, paperInfo: PaperInfo):
        if self.source == "s2":
            paperInfo = self._s2_download(paperInfo)
        elif self.source.lower() == "arxiv":
            paperInfo = self._arX_download(paperInfo)

        return paperInfo

    def _s2_download(self, paperInfo: PaperInfo):
        try:
            extensions = ("*.pdf", "*.crdownload")
            paper_download_dir = os.path.join(
                self.downloads_dir, paperInfo["paperId"])
            os.makedirs(paper_download_dir, exist_ok=True)

            self.queue.append(paperInfo["paperId"])

            # Set Chrome's download directory dynamically
            self.driver.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": paper_download_dir})

            clean_url = re.sub(r"^http://", "https://",
                               paperInfo["openAccessPdf"])
            filename = None
            self.driver.set_page_load_timeout(self.timeout_loadpage)
            self.driver.get(clean_url)
            end_time_startdw = time.time() + self.timeout_startdw

            download_status = "download too long"

            if (any(marker.lower() in self.driver.page_source.lower()
                    for marker in self.block_markers)
                    or "crawlprevention" in self.driver.current_url
                    or "europepmc.org/backend" in self.driver.current_url
                    or any(marker.lower() in self.driver.title.lower()
                           for marker in self.error_markers)):
                download_status = self.driver.current_url
            else:
                while time.time() < end_time_startdw:
                    filename = [
                        fname
                        for ext in extensions
                        for fname in glob.glob(os.path.join(paper_download_dir, ext))
                    ]
                    if filename:
                        download_status = "downloading"
                        break
                    else:
                        time.sleep(0.2)

        except Exception as e:
            download_status = "broken link?"

        paperInfo["pdf_status"] = download_status

        return paperInfo

    def is_download_finished(self):
        if os.path.isdir(self.downloads_dir):
            for paperId in self.queue:
                dir_path = os.path.join(self.downloads_dir, paperId)
                if os.path.isdir(dir_path):
                    try:
                        if (glob.glob(os.path.join(dir_path, "*.crdownload"))
                                or time.time() - os.path.getmtime(dir_path) < self.timeout_loadpage):
                            return False
                    except Exception:
                        pass
        return True

    def _arX_download(self, paperInfo: PaperInfo):
        extensions = ("*.pdf")
        paper_download_dir = os.path.join(
            self.downloads_dir, paperInfo["paperId"])
        os.makedirs(paper_download_dir, exist_ok=True)

        self.queue.append(paperInfo["paperId"])
        try:
            subprocess.run(["gsutil", "cp", paperInfo["openAccessPdf"],
                            str(paper_download_dir)], check=True)
            endtime = time.time() + self.timeout_loadpage
            download_status = "download too long"
            while time.time() < endtime:
                filename = [fname
                            for ext in extensions
                            for fname in glob.glob(os.path.join(paper_download_dir, ext))
                            ]
                if filename:
                    download_status = "downloaded"
                    break
                else:
                    time.sleep(0.2)
        except Exception as e:
            download_status = "broken link?"

        paperInfo["pdf_status"] = download_status

        return paperInfo


def close_all_chrome_sessions(verbose: bool = True):
    nkill = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info["name"] and (
            "chrome" in proc.info["name"].lower()
                or "chromium" in proc.info["name"].lower()):
            try:
                proc.kill()
                nkill += 1
            except Exception as e:
                logger.warning(
                    f"Could not close process {proc.info['pid']}: {e}")

    if verbose:
        logger.info(f"Closed {nkill} Chrome processes")


def download_worker_init(
        downloads_dir: str,
        source: str = "s2",
        timeout_loadpage: Optional[float] = 15,
        timeout_startdw: Optional[float] = 5):
    global downloader
    downloader = PDFdownloader(downloads_dir,
                               source=source,
                               headless=False,
                               worker_id=os.getpid(),
                               timeout_loadpage=timeout_loadpage,
                               timeout_startdw=timeout_startdw)

    atexit.register(download_worker_close)


def download_worker_close():
    global downloader
    downloader.close()


def download_PDF_workers(papers):
    global downloader
    output = [downloader.download(paper) for paper in papers]
    while not downloader.is_download_finished():
        time.sleep(1)
    downloader.reset_queue()
    return output


def collect_downloaded_ids(directory: str):
    folder_list = os.listdir(directory)
    paper_downloaded = []
    for paper_id in folder_list:
        if os.path.isdir(os.path.join(directory, paper_id)):
            pdf_path = glob.glob(os.path.join(directory, paper_id, "*.pdf"))
            if pdf_path:
                paper_downloaded.append(paper_id)

    return paper_downloaded


def download_PDFs(
        years: Optional[int] = None,
        filters: Optional[List[Any] |
                          List[List[Any]]] = None,
        n_jobs: Optional[int] = pdf_config['n_jobs_download'],
        timeout_loadpage: Optional[float] = pdf_config['timeout_loadpage'],
        timeout_startdw: Optional[float] = pdf_config['timeout_startdw'],
        save_pdfs_to: Optional[str] = pdf_config['save_pdfs_to']):

    years = [datetime.now().year] if years is None else years
    years = [years] if not isinstance(years, list) else years

    if save_pdfs_to is not None:
        downloads_dir_base = save_pdfs_to
    else:
        downloads_dir_base = config.DATA_PATH / "tmp" / "download"

    for year in years:
        downloads_dir = downloads_dir_base / str(year)
        os.makedirs(downloads_dir, exist_ok=True)

    # filters = [['abstract','==','None']]
    n_jobs = min(n_jobs, multiprocessing.cpu_count() // 3)
    results = []
    for year in years:
        downloads_dir = downloads_dir_base / str(year)

        logger.info(f"Downloading pdfs for year {year}")
        paperdata = load_dataset(subsets=year,
                                 data_types=["metadata", "text"])

        if filters is not None:
            paperdata_filt = load_dataset(subsets=year,
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

        paperInfo_s = {"http": paperInfo[paperInfo["openAccessPdf"].str.startswith("http")],
                       "gs": paperInfo[paperInfo["openAccessPdf"].str.startswith("gs:")], }
        external_sources = {"http", "gs"}
        for source in external_sources:
            if source is not None:
                paperInfo_s = paperInfo[paperInfo["openAccessPdf"].str.startswith(
                    source)]
            else:
                paperInfo_s = paperInfo

            if not paperInfo_s.empty:
                minibatch_size = 20  # 100
                batch_size = 3000
                Npapers = 3000  # len(paperInfo_s)
                tqdm = get_tqdm()
                with tqdm(range(math.ceil(Npapers / (batch_size + 1))),
                          desc=f"Downloading {Npapers} papers from {year}") as pbar:
                    try:
                        for i in pbar:
                            close_all_chrome_sessions()
                            print(downloads_dir)
                            endtime = time.time() + timeout_loadpage * batch_size

                            batch = paperInfo_s.iloc[i *
                                                     batch_size: (i+1)*batch_size].to_dict(orient="records")

                            Nbatch = len(batch)
                            minibatches = [batch[k * minibatch_size: min(Nbatch, (k + 1) * minibatch_size)]
                                           for k in range(Nbatch // minibatch_size + 1)]

                            minibatches = [b for b in minibatches
                                           if len(b) > 0 and b[0] is not None]

                            if minibatches:
                                pdf_downloaded, pdf_new = [], []
                                download_pool = multiprocessing.Pool(
                                    processes=n_jobs,
                                    initializer=partial(download_worker_init,
                                                        downloads_dir=downloads_dir,
                                                        timeout_loadpage=timeout_loadpage,
                                                        timeout_startdw=timeout_startdw))

                                with download_pool:
                                    async_dwload = [download_pool.apply_async(download_PDF_workers, (b,))
                                                    for b in minibatches]
                                    pending_dwload = set(async_dwload)

                                    while pending_dwload:
                                        if len(pdf_new) == 0 and time.time() > endtime:
                                            logger.info(
                                                f"{len(pending_dwload)} downloads out of {len(minibatches)} timed out. Terminating the pool.")

                                            download_pool.terminate()
                                            download_pool.join()
                                            break

                                        pdf_dw = collect_downloaded_ids(
                                            downloads_dir)
                                        pdf_new = [
                                            pdf for pdf in pdf_dw if pdf not in pdf_downloaded]

                                        for task in list(pending_dwload):
                                            if task.ready():
                                                res_dw = task.get()
                                                results.append(res_dw)
                                                pending_dwload.remove(task)

                                        if pending_dwload:
                                            pbar.set_postfix(
                                                {"download tasks": f"{len(pending_dwload)} left"})

                                            time.sleep(1)

                                        pdf_downloaded = pdf_dw
                                        if len(pdf_new) > 0:
                                            endtime = time.time() + 3*timeout_loadpage

                                close_all_chrome_sessions()

                    except Exception as e:
                        pbar.close()
                        raise e

    return results
