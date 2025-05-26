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

from asxai.logger import get_logger
from asxai.utils import get_tqdm, running_inside_docker
import psutil

logger = get_logger(__name__, level=config.LOG_LEVEL)

params = load_params()
pdf_config = params["pdf"]


class PaperInfo(TypedDict):
    paperId: str
    openAccessPdf: str


class PDFdownloader:
    def __init__(self,
                 downloads_dir: str,
                 source: str = "http",
                 headless: Optional[bool] = False,
                 worker_id: Optional[int] = 0,
                 timeout_loadpage: Optional[float] = 15,
                 timeout_startdw: Optional[float] = 5):
        self.source = source
        self.queue = []
        self.headless = headless
        self.worker_id = worker_id

        hostname = "selenium-hub" if running_inside_docker() else "localhost"

        self.downloads_dir = downloads_dir
        year_dir = pathlib.Path(downloads_dir).stem
        self.chrome_downloads_dir = "/home/seluser/Downloads/" + year_dir

        self.selenium_hub_url = os.getenv(
            "SELENIUM_HUB_URL", f"http://{hostname}:4444/wd/hub")

        temp_dir = tempfile.gettempdir()
        # self.user_data_dir = tempfile.mkdtemp(
        #     prefix=f"chrome-profile-{self.worker_id}-", dir=temp_dir)
        if self.source == "http":
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
        # chromedriver_path = shutil.which("chromedriver")
        # if not chromedriver_path:
        #     raise FileNotFoundError(
        #         "chromedriver not found in your PATH. Please install it or add it to PATH.")
        # service = Service(executable_path=chromedriver_path)

        options = webdriver.ChromeOptions()
        # options.binary_location = "/usr/bin/chromium-browser"
        # options.add_argument(f"--user-data-dir={str(self.user_data_dir)}")
        # options.add_argument("--disable-gpu")
        # options.add_argument("--no-sandbox")
        # options.add_argument("--disable-dev-shm-usage")
        if headless:
            options.add_argument("--headless=new")
        # prefs = {
        #     "plugins.always_open_pdf_externally": True,
        #     "download.prompt_for_download": False,
        #     "download.default_directory": str(self.chrome_downloads_dir),
        #     "download.directory_upgrade": True}

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        prefs = {
            "plugins.always_open_pdf_externally": True,
            "download.prompt_for_download": False,
            "download.default_directory": str(self.chrome_downloads_dir)}
        options.add_experimental_option("prefs", prefs)
        # chromedriver_path = shutil.which("chromedriver")

        waitforSession = 0
        max_retries = 3
        retry_delay = 3
        while waitforSession < max_retries:
            try:
                # driver = webdriver.Chrome(service=service, options=options)
                driver = webdriver.Remote(command_executor=self.selenium_hub_url,
                                          options=options)
                RemoteConnection.set_timeout(10)
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
            try:
                time.sleep(1)
                self.driver.quit()
                # shutil.rmtree(self.user_data_dir, ignore_errors=True)
            finally:
                self.driver = None

    def reset_queue(self):
        self.queue = []

    def download(self, paperInfo: PaperInfo):
        if self.source == "http":
            paperInfo = self._http_download(paperInfo)
        elif self.source.lower() == "gs":
            paperInfo = self._gs_download(paperInfo)

        return paperInfo

    def _http_download(self, paperInfo: PaperInfo):
        try:
            extensions = ("*.pdf", "*.crdownload")
            paper_dir = os.path.join(
                self.downloads_dir, paperInfo["paperId"])
            os.makedirs(paper_dir, mode=777, exist_ok=True)
            os.chmod(paper_dir, 0o777)

            chrome_paper_dir = os.path.join(
                self.chrome_downloads_dir, paperInfo["paperId"])

            self.queue.append(paperInfo["paperId"])

            # Set Chrome's download directory dynamically
            self.driver.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": chrome_paper_dir})

            clean_url = re.sub(r"^http://", "https://",
                               paperInfo["openAccessPdf"])
            filename = None
            self.driver.set_page_load_timeout(self.timeout_loadpage)

            # script = f"""
            #         const link = document.createElement('a');
            #         link.href = '{clean_url}';
            #         document.body.appendChild(link);
            #         link.click();
            #         document.body.removeChild(link);
            #         """

            # self.driver.execute_script(script)

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
                        for fname in glob.glob(os.path.join(paper_dir, ext))
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

    def _gs_download(self, paperInfo: PaperInfo):
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
                break

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
        source: str = "http",
        timeout_loadpage: Optional[float] = 15,
        timeout_startdw: Optional[float] = 5,
        run_headless: Optional[float] = False):
    global downloader
    downloader = PDFdownloader(downloads_dir,
                               source=source,
                               headless=run_headless,
                               worker_id=os.getpid(),
                               timeout_loadpage=timeout_loadpage,
                               timeout_startdw=timeout_startdw)

    atexit.register(download_worker_close)


def download_worker_close():
    global downloader
    downloader.close()


def download_PDF_workers(papers):
    global downloader
    try:
        output = [downloader.download(paper) for paper in papers]
        while not downloader.is_download_finished():
            time.sleep(1)
        downloader.reset_queue()
        return output
    finally:
        downloader.reset_queue()
        downloader.close()


def downloaded_year_done(directory: str):
    folder_list = os.listdir(directory)
    return 'done' in folder_list


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
        paperdata: pd.DataFrame,
        year: int,
        n_jobs: Optional[int] = pdf_config['n_jobs_download'],
        timeout_loadpage: Optional[float] = pdf_config['timeout_loadpage'],
        timeout_startdw: Optional[float] = pdf_config['timeout_startdw'],
        save_pdfs_to: Optional[str] = pdf_config['save_pdfs_to'],
        run_headless: Optional[bool] = pdf_config['run_headless']):

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
    paperInfo = pd.merge(paperdata["text"][["paperId", "openAccessPdf"]],
                         paperdata["metadata"][[
                             "paperId", "authorName"]],
                         on="paperId",
                         how="inner")

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
            with tqdm(range(math.ceil(Npapers / (batch_size + 1))),
                      desc=f"Downloading {Npapers} papers from {year}") as pbar:
                try:
                    for i in pbar:
                        # close_all_chrome_sessions()
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
                                                    source=source,
                                                    downloads_dir=downloads_dir,
                                                    timeout_loadpage=timeout_loadpage,
                                                    timeout_startdw=timeout_startdw,
                                                    run_headless=run_headless))

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

                            # close_all_chrome_sessions()
                            download_pool.close()
                            download_pool.join()

                except Exception as e:
                    pbar.close()
                    raise e

    os.mkdir(os.path.join(downloads_dir, 'done'))
