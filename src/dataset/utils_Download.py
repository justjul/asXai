import os
import glob
import shutil
import tempfile
import re
import time
from typing import Optional, TypedDict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

import logging
from src.logger import get_logger
import psutil

logger = get_logger(__name__, level=logging.INFO)


class PaperInfo(TypedDict):
    paperId: str
    openAccessPdf: str


class PDFdownloader:
    def __init__(self,
                 downloads_dir: str,
                 headless: Optional[bool] = False,
                 worker_id: Optional[int] = 0,
                 timeout_loadpage: Optional[float] = 15,
                 timeout_startdw: Optional[float] = 5):
        self.downloads_dir = downloads_dir
        self.queue = []
        self.headless = headless
        self.worker_id = worker_id

        temp_dir = tempfile.gettempdir()
        for dirname in os.listdir(temp_dir):
            if "chrome-profile-" in dirname:
                shutil.rmtree(os.path.join(temp_dir, dirname))

        self.user_data_dir = tempfile.mkdtemp(
            prefix="chrome-profile-{self.worker_id}-", dir=temp_dir)

        self.driver = self._init_driver(headless)

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

    def _init_driver(self, headless: bool = False):
        chromedriver_path = shutil.which("chromedriver")
        if not chromedriver_path:
            raise FileNotFoundError(
                "chromedriver not found in your PATH. Please install it or add it to PATH.")
        service = Service(executable_path=chromedriver_path)

        options = webdriver.ChromeOptions()
        options.add_argument(f"--user-data-dir={self.user_data_dir}")
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
        self.driver.quit()
        shutil.rmtree(self.user_data_dir, ignore_errors=True)

    def reset_queue(self):
        self.queue = []

    def download(self, paperInfo: PaperInfo):
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
                    if (glob.glob(os.path.join(dir_path, "*.crdownload"))
                            or time.time() - os.path.getmtime(dir_path) < self.timeout_loadpage):
                        return False
        return True


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
