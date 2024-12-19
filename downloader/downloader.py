import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from zipfile import BadZipFile

import requests
from tqdm import tqdm

KIB = 2 ** 10

import logging
from logging import Logger, StreamHandler, Formatter


def create_default_logger(instance: object = None) -> Logger:
    """
    Creates and returns a default logger.

    Args:
        instance (object, optional): An object instance to use in naming the logger.

    Returns:
        Logger: A configured logger instance.
    """
    if instance:
        logger_name = f"{instance.__class__.__name__}_{id(instance)}"
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger(__name__)

    logger.setLevel("INFO")

    if not logger.handlers:
        console_handler = StreamHandler()
        console_handler.setFormatter(Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(console_handler)

    return logger


class Downloader:
    """
    A class to handle downloading and extracting ZIP files from a URL.

    Attributes:
        root (str): The root directory where files will be saved.
        url (str): The URL to download the file from.
        zip_filename (str): The name of the zip file to be saved.
        zip_path (Path): The path to the zip file.
        extract_path (Path): The path to the extracted files.
        reextract (bool): Whether to re-extract the files.
        redownload (bool): Whether to re-download the file.
        max_fetch_retries (int): The maximum number of retries for download.
        retry_delay (int): The delay between retry attempts in seconds.
        logger (Logger): The logger instance used for logging events.

    Methods:
        is_downloaded (bool): Checks if the ZIP file has been downloaded.
        is_extracted (bool): Checks if the ZIP file has been extracted.
        download() -> None: Downloads the ZIP file from the URL.
        extract() -> None: Extracts the downloaded ZIP file.
        __call__(on_complete: callable = None) -> None: Initiates the download and extraction process.
    """

    def __init__(self,
                 root: str,
                 url: str,
                 zip_filename: str = None,
                 redownload: bool = False,
                 reextract: bool = False,
                 max_fetch_retries: int = 3,
                 fetch_retry_delay: int = 2,
                 logger: Logger = None,
                 ):
        """
        Initializes the Downloader instance.

        Args:
            root (str): The directory where files will be saved.
            url (str): The URL to download the ZIP file from.
            zip_filename (str, optional): The name of the zip file. Defaults to None.
            redownload (bool, optional): Whether to redownload the file if it exists. Defaults to False.
            reextract (bool, optional): Whether to re-extract the files if they exist. Defaults to False.
            max_fetch_retries (int, optional): The number of download retry attempts. Defaults to 3.
            fetch_retry_delay (int, optional): The delay (in seconds) between download retries. Defaults to 2.
            logger (Logger, optional): A custom logger instance. Defaults to None.
        """
        self._root = Path(root).resolve()
        self.url = url
        self.zip_filename = zip_filename or Path(urlparse(self.url).path).name
        self.zip_path = self._root / self.zip_filename
        self.extract_path = self.zip_path.with_suffix("")

        self.reextract = reextract
        self.redownload = redownload

        self.max_fetch_retries = max_fetch_retries
        self.retry_delay = fetch_retry_delay
        self.logger = logger or create_default_logger(self)

    @property
    def is_downloaded(self) -> bool:
        """
        Checks if the ZIP file has been downloaded.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return self.zip_path.exists()

    @property
    def is_extracted(self) -> bool:
        """
        Checks if the ZIP file has been extracted.

        Returns:
            bool: True if the extraction directory exists, False otherwise.
        """
        return self.extract_path.exists()

    def download(self) -> None:
        """
        Downloads the ZIP file from the URL.

        Raises:
            requests.exceptions.RequestException: If the download fails after multiple retries.
        """
        self.logger.info(f"Downloading from {self.url}...")
        self._root.mkdir(parents=True, exist_ok=True)
        self._save_content()
        self.logger.info(f"Downloaded successfully to {self.zip_path}.")

    def _save_content(self) -> None:
        """
        Saves the content from the URL to the specified file path using streaming.

        Args:
            None

        Raises:
            requests.exceptions.RequestException: If the download fails.
        """
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        with self.zip_path.open("wb") as f:
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1000) as progress_bar:
                for chunk in response.iter_content(chunk_size=100 * KIB):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

    def extract(self) -> None:
        """
        Extracts the downloaded ZIP file to the specified directory.

        Raises:
            zipfile.BadZipFile: If the ZIP file is corrupted.
        """
        self.logger.info(f"Unzipping {self.zip_path}...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            top_level_dirs = {Path(x).parts[0] for x in zip_ref.namelist()}

            if len(top_level_dirs) == 1:
                top_level_dir = next(iter(top_level_dirs))
                self.extract_path = self._root / top_level_dir
                self.logger.info(f"Extracting into {self.extract_path}...")

            total_files = len(zip_ref.namelist())
            with tqdm(total=total_files, unit='file') as progress:
                for file in zip_ref.namelist():
                    destination = self._root if len(top_level_dirs) == 1 else self.extract_path
                    zip_ref.extract(file, destination)
                    progress.update(1)

        self.logger.info(f"Extracted successfully to {self.extract_path}.")

    def __call__(self, on_complete: callable = None):
        """
        Downloads and extracts the ZIP file, handling retries for both steps.

        Args:
            on_complete (callable, optional): A callback function to call after the process is complete.
        """
        if self.redownload or not self.is_downloaded:
            for retry in range(self.max_fetch_retries):
                try:
                    self.download()
                    break
                except requests.RequestException as err:
                    self.logger.warning(f"Download attempt {retry + 1} failed: {err}")
                    time.sleep(self.retry_delay)
            else:
                self.logger.error("Failed to download after multiple attempts.")
                return
        else:
            self.logger.info(f"Dataset already exists at {self.zip_path}, skipping download.")

        if self.reextract or not self.is_extracted:
            self.logger.info("Extracting...")
            for retry in range(self.max_fetch_retries):
                try:
                    self.extract()
                    break
                except BadZipFile:
                    self.logger.warning("BadZipFile detected. Redownloading the file...")
                    self.zip_path.unlink(missing_ok=True)
        else:
            self.logger.info(f"Dataset already extracted at {self.extract_path}, skipping extraction.")

        if on_complete is not None:
            on_complete()
