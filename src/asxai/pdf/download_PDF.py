"""
asxai.pdf.download_PDF module

Provides functionality to download research paper PDFs using two methods:
1. HTTP download via Selenium WebDriver.
2. Google Storage download via gsutil (for arXiv pdfs).

Supports concurrent downloads using multiprocessing pools, dynamic download directory management,
queue tracking, and cleanup utilities for Chrome/Chromium processes.
"""

import config
from asxai.utils import load_params
from functools import partial
import multiprocessing
from typing import List, Optional, Any, TypedDict
import pandas as pd
import math
import os
import pathlib
import glob
import tempfile
import re
import time
from typing import Optional, TypedDict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.remote_connection import RemoteConnection
import subprocess

import atexit
import psutil

from asxai.logger import get_logger
from asxai.utils import get_tqdm, running_inside_docker

# Initialize logger for this module
logger = get_logger(__name__, level=config.LOG_LEVEL)

# Load configuration parameters for PDF downloading
params = load_params()
pdf_config = params["pdf"]


class PaperInfo(TypedDict):
    """
    TypedDict to represent minimal information needed to download a paper.

    Attributes:
        paperId: Unique identifier for the paper.
        openAccessPdf: URL or URI to the PDF file.
    """
    paperId: str
    openAccessPdf: str


class PDFdownloader:
    """
    Handles downloading of single or multiple PDFs via HTTP or Google Storage (gsutil).
    Uses Selenium for HTTP downloads to handle browser-based interactions.
    """

    def __init__(
        self,
        downloads_dir: str,
        source: str = "http",
        headless: Optional[bool] = False,
        worker_id: Optional[int] = 0,
        timeout_loadpage: Optional[float] = 15,
        timeout_startdw: Optional[float] = 5
    ):
        """
        Handles downloading PDFs either via HTTP (through Selenium) or via Google Storage (gsutil).

        Attributes:
            source: 'http' or 'gs', indicating download method.
            downloads_dir: Filesystem path where papers will be saved.
            headless: Run Chrome in headless mode if True.
            worker_id: Identifier for multi-process workers.
            timeout_loadpage: Max seconds to wait for page load.
            timeout_startdw: Max seconds to wait for download to begin.
        """
        self.source = source
        self.queue = []  # Track pending paperIds
        self.headless = headless
        self.worker_id = worker_id

        # Determine Selenium hub hostname (Docker vs local)
        hostname = "selenium-hub" if running_inside_docker() else "localhost"

        # Configure local and Chrome container download paths
        self.downloads_dir = downloads_dir
        year_dir = pathlib.Path(downloads_dir).stem
        # Chrome container's default download directory for seluser
        self.chrome_downloads_dir = "/home/seluser/Downloads/" + year_dir

        # URL for remote Selenium WebDriver
        self.selenium_hub_url = os.getenv(
            "SELENIUM_HUB_URL",
            f"http://{hostname}:4444/wd/hub"
        )

        self.timeout_startdw = timeout_startdw
        self.timeout_loadpage = timeout_loadpage

        # Markers indicating blocked/error pages
        self.block_markers = {
            "captcha",
            "verify you are human",
            "not a robot",
            "forbidden",
            "error: 500",
            "this site can’t be reached"
        }
        self.error_markers = {
            "just a moment",
            "privacy error",
            "validate user",
            "error 500"
        }

        # Initialize Selenium driver only for HTTP source
        self.driver = None
        if self.source == "http":
            self.driver = self._s2_init_driver(headless)

    def _s2_init_driver(self, headless: bool = False):
        """
        Initialize a Selenium Remote WebDriver session connected to Selenium hub.
        Retries on failure up to 3 times.
        """

        options = webdriver.ChromeOptions()
        # Run in headless mode if specified
        if headless:
            options.add_argument("--headless=new")
        # Recommended flags for container environments
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # Set Chrome preferences for direct PDF download without prompt
        prefs = {
            "plugins.always_open_pdf_externally": True,
            "download.prompt_for_download": False,
            "download.default_directory": str(self.chrome_downloads_dir)}
        options.add_experimental_option("prefs", prefs)

        # Attempt to connect to remote WebDriver
        retry_count = 0
        max_retries = 3
        retry_delay = 3
        conn_timeout = 60  # seconds for session creation
        while retry_count < max_retries:
            try:
                driver = webdriver.Remote(
                    command_executor=self.selenium_hub_url,
                    options=options
                )
                RemoteConnection.set_timeout(conn_timeout)
                # Short delay to ensure session stable
                time.sleep(0.5)
                return driver
            except Exception as e:
                logger.warning(
                    f"Session creation failed ({retry_count+1}/{max_retries}): {e}"
                )
                time.sleep(retry_delay)
                retry_count += 1

        # If all retries failed, log error and leave driver as None
        logger.error(
            f"Failed to create a Selenium session after {max_retries} attempts"
        )
        return None

    def close(self) -> None:
        """
        Gracefully close the Selenium WebDriver session.
        """
        if self.driver is not None:
            try:
                time.sleep(1)
                self.driver.quit()
                # shutil.rmtree(self.user_data_dir, ignore_errors=True)
            finally:
                self.driver = None

    def reset_queue(self) -> None:
        """
        Clear the internal download queue tracking.
        """
        self.queue = []

    def download(self, paperInfo: PaperInfo):
        """
        Download a single paper based on the configured source.
        Chooses HTTP or GS method.

        Attributes:
            paperInfo: Dictionary with paperId and openAccessPdf link
        Return: 
            Updated paperInfo with 'status' field
        """
        if self.source == "http":
            paperInfo = self._http_download(paperInfo)
        elif self.source.lower() == "gs":
            paperInfo = self._gs_download(paperInfo)

        return paperInfo

    def _http_download(self, paperInfo: PaperInfo):
        """
        Download PDF via HTTP using Selenium browser automation.
        Handles dynamic download path, retries blocked pages, and status tracking.
        """
        try:
            # Create target directories
            extensions = ("*.pdf", "*.crdownload")
            paper_dir = os.path.join(
                self.downloads_dir, paperInfo["paperId"])
            os.makedirs(paper_dir, mode=777, exist_ok=True)
            os.chmod(paper_dir, 0o777)

            chrome_paper_dir = os.path.join(
                self.chrome_downloads_dir, paperInfo["paperId"])

            self.queue.append(paperInfo["paperId"])

            # Ensure driver is initialized
            if self.driver is None:
                self.driver = self._s2_init_driver(self.headless)

            # Set Chrome's download directory dynamically
            self.driver.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": chrome_paper_dir})

            # Standardize URL to HTTPS
            clean_url = re.sub(r"^http://", "https://",
                               paperInfo["openAccessPdf"])
            filename = None
            self.driver.set_page_load_timeout(self.timeout_loadpage)

            # Visit URL to trigger download
            self.driver.get(clean_url)

            # Monitor for download start or blocking errors
            end_time_startdw = time.time() + self.timeout_startdw
            download_status = "download too long"

            # If a block marker is detected, abort early
            if (any(marker.lower() in self.driver.page_source.lower()
                    for marker in self.block_markers)
                    or "crawlprevention" in self.driver.current_url
                    or "europepmc.org/backend" in self.driver.current_url
                    or any(marker.lower() in self.driver.title.lower()
                           for marker in self.error_markers)):
                download_status = self.driver.current_url
            else:
                # Poll filesystem for new PDF or .crdownload files
                while time.time() < end_time_startdw:
                    filename = [
                        fname
                        for ext in extensions
                        for fname in glob.glob(os.path.join(paper_dir, ext))
                    ]
                    if filename:
                        download_status = "downloading"
                        break
                    else:
                        time.sleep(0.2)

        except Exception as e:
            logger.warning(f"Unexpected error: {e}")
            download_status = "broken link?"

        paperInfo["status"] = download_status

        return paperInfo

    def is_download_finished(self) -> bool:
        """
        Check if all queued downloads have completed (no .crdownload files remain).
        """
        if os.path.isdir(self.downloads_dir):
            for paperId in self.queue:
                dir_path = os.path.join(self.downloads_dir, paperId)
                if os.path.isdir(dir_path):
                    # If any temp file exists or folder recently modified, still downloading
                    try:
                        if (glob.glob(os.path.join(dir_path, "*.crdownload"))
                                or time.time() - os.path.getmtime(dir_path) < self.timeout_loadpage):
                            return False
                    except Exception:
                        pass
        return True

    def _gs_download(self, paperInfo: PaperInfo):
        """
        Download PDF from Google Storage using `gsutil cp`.

        Retries on version mismatches and network errors.
        """
        extensions = ("*.pdf")
        paper_download_dir = os.path.join(
            self.downloads_dir, paperInfo["paperId"])
        os.makedirs(paper_download_dir, mode=777, exist_ok=True)
        os.chmod(paper_download_dir, 0o777)

        self.queue.append(paperInfo["paperId"])

        pdf_url = paperInfo["openAccessPdf"]
        version_match = re.search(r"v(\d+)\.pdf$", pdf_url)

        download_status = "broken link?"
        n_attempts = 3

        for retry in range(n_attempts + 1):
            try:
                # If retrying, adjust version number
                if retry > 0 and version_match:
                    version = int(version_match.group(1)) - retry
                    if version <= 0:
                        break
                    pdf_url = re.sub(r"v\d+\.pdf$", f"v{version}.pdf", pdf_url)

                subprocess.run(["gsutil", "-q", "cp",
                                pdf_url, str(paper_download_dir)], check=True)
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
                    time.sleep(0.2)

                if download_status == "downloaded":
                    break
            except subprocess.CalledProcessError:
                continue
            except Exception as e:
                logger.warning(f"Unexpected error: {e}")
                download_status = "broken link?"
                break

        paperInfo["status"] = download_status

        return paperInfo


def close_all_chrome_sessions(verbose: bool = True):
    """
    Kill all running Chrome/Chromium processes to free resources.
    """
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
    source: str = "http",
    timeout_loadpage: Optional[float] = 15,
    timeout_startdw: Optional[float] = 5,
    run_headless: Optional[float] = False
) -> None:
    """
    Initialize a global PDFdownloader instance for each worker.
    Registered with multiprocessing.Pool initializer.
    """
    global downloader
    downloader = PDFdownloader(downloads_dir,
                               source=source,
                               headless=run_headless,
                               worker_id=os.getpid(),
                               timeout_loadpage=timeout_loadpage,
                               timeout_startdw=timeout_startdw)

    # Ensure driver is closed on worker exit
    atexit.register(download_worker_close)


def download_worker_close():
    """
    Cleanup function to close the downloader at worker exit.
    """
    global downloader
    downloader.close()


def download_PDF_workers(papers):
    """
    Download a batch of papers in a worker process.

    Waits for all downloads to complete before returning statuses.
    """
    global downloader
    try:
        output = [downloader.download(paper) for paper in papers]
        # Wait until filesystem indicates all downloads finished
        while not downloader.is_download_finished():
            time.sleep(1)
        downloader.reset_queue()
        return output
    finally:
        downloader.reset_queue()
        downloader.close()


def downloaded_year_done(directory: str) -> bool:
    """
    Check if a 'done' marker folder exists, indicating completion.
    """
    folder_list = os.listdir(directory)
    return 'done' in folder_list


def collect_downloaded_ids(directory: str) -> List[str]:
    """
    List paper IDs which have a .pdf file present.
    """
    folder_list = os.listdir(directory)
    paper_downloaded = []
    for paper_id in folder_list:
        if os.path.isdir(os.path.join(directory, paper_id)):
            pdf_path = glob.glob(os.path.join(directory, paper_id, "*.pdf"))
            if pdf_path:
                paper_downloaded.append(paper_id)

    return paper_downloaded


def download_PDFs(
    paperdata: pd.DataFrame,
    year: int,
    n_jobs: Optional[int] = pdf_config['n_jobs_download'],
    timeout_loadpage: Optional[float] = pdf_config['timeout_loadpage'],
    timeout_startdw: Optional[float] = pdf_config['timeout_startdw'],
    save_pdfs_to: Optional[str] = pdf_config['save_pdfs_to'],
    run_headless: Optional[bool] = pdf_config['run_headless']
):
    """
    Orchestrate parallel downloading of PDFs for a given year.

    Arguments:
        paperdata: DataFrame with 'text' and 'metadata' containing paperId and URLs.
        year: Publication year of papers.
        n_jobs: Number of parallel worker processes.
        save_pdfs_to: Base directory to save PDFs; uses config.TMP_PATH if None.
        run_headless: Launch Chrome in headless mode.

    Creates subdirectories per paperId and a 'done' marker on completion.
    """
    # Determine base downloads directory
    if save_pdfs_to is not None:
        downloads_dir_base = save_pdfs_to
    else:
        downloads_dir_base = config.TMP_PATH / "downloads"

    n_jobs = min(n_jobs, 20)
    results = []
    downloads_dir = downloads_dir_base / str(year)
    os.makedirs(downloads_dir, mode=777, exist_ok=True)
    os.chmod(downloads_dir, 0o777)

    logger.info(f"Downloading pdfs for year {year}")
    # Merge necessary columns into a single table
    paperInfo = pd.merge(
        paperdata["text"][["paperId", "openAccessPdf"]],
        paperdata["metadata"][["paperId", "authorName"]],
        on="paperId",
        how="inner"
    )

    paperInfo_s = {"http": paperInfo[paperInfo["openAccessPdf"].str.startswith("http")],
                   "gs": paperInfo[paperInfo["openAccessPdf"].str.startswith("gs:")], }
    external_sources = ["http", "gs"]
    for source in external_sources:
        if source is not None:
            paperInfo_s = paperInfo[paperInfo["openAccessPdf"].str.startswith(
                source)]
        else:
            paperInfo_s = paperInfo

        if not paperInfo_s.empty:
            minibatch_size = 20  # 100
            batch_size = minibatch_size * n_jobs  # 512
            Npapers = len(paperInfo_s)
            tqdm = get_tqdm()
            # Iterate over batches
            with tqdm(range(math.ceil(Npapers / (batch_size + 1))),
                      desc=f"Downloading {Npapers} papers from {year}") as pbar:
                try:
                    for i in pbar:
                        print(downloads_dir)
                        endtime = time.time() + timeout_loadpage * batch_size

                        batch = paperInfo_s.iloc[i *
                                                 batch_size: (i+1)*batch_size].to_dict(orient="records")

                        Nbatch = len(batch)
                        # Break into minibatches for each process
                        minibatches = [batch[k * minibatch_size: min(Nbatch, (k + 1) * minibatch_size)]
                                       for k in range(Nbatch // minibatch_size + 1)]

                        minibatches = [b for b in minibatches
                                       if len(b) > 0 and b[0] is not None]

                        if minibatches:
                            pdf_downloaded, pdf_new = [], []
                            # Launch worker pool
                            download_pool = multiprocessing.Pool(
                                processes=n_jobs,
                                initializer=partial(
                                    download_worker_init,
                                    source=source,
                                    downloads_dir=downloads_dir,
                                    timeout_loadpage=timeout_loadpage,
                                    timeout_startdw=timeout_startdw,
                                    run_headless=run_headless
                                )
                            )

                            with download_pool:
                                async_dwload = [download_pool.apply_async(download_PDF_workers, (b,))
                                                for b in minibatches]
                                # Monitor progress until done or timeout
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
                                    # Check for ready and timed-out tasks
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

                            # close_all_chrome_sessions()
                            download_pool.close()
                            download_pool.join()

                except Exception as e:
                    logger.error(f"Fatal error: {e}")
                    # pbar.close()
                    # raise e

    # Mark completion
    os.mkdir(os.path.join(downloads_dir, 'done'))
