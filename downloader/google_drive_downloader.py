from logging import Logger

import gdown

from downloader import Downloader


class GoogleDriveDownloader(Downloader):
    """
    A class to handle downloading ZIP files from Google Drive.

    This class extends the `Downloader` class and specifically handles the
    process of downloading ZIP files from a Google Drive URL using the file ID.

    Attributes:
        root (str): The root directory where files will be saved.
        file_id (str): The Google Drive file ID used to generate the download URL.
        zip_filename (str): The name of the zip file to be saved.
        zip_path (Path): The path to the zip file.
        extract_path (Path): The path to the extracted files.
        reextract (bool): Whether to re-extract the files.
        redownload (bool): Whether to re-download the file.
        logger (Logger): The logger instance used for logging events.

    Methods:
        _save_content() -> None: Downloads the file from Google Drive and saves it to the zip path.
    """

    def __init__(self,
                 root: str,
                 file_id: str,
                 redownload: bool = False,
                 reextract: bool = False,
                 zip_filename: str = None,
                 max_fetch_retries=3,
                 max_fetch_retry_delay=2,
                 logger: Logger = None,
                 ):
        """
        Initializes the GoogleDriveDownloader instance.

        Args:
            root (str): The directory where files will be saved.
            file_id (str): The Google Drive file ID used to generate the download URL.
            redownload (bool, optional): Whether to redownload the file if it exists. Defaults to False.
            reextract (bool, optional): Whether to re-extract the files if they exist. Defaults to False.
            zip_filename (str, optional): The name of the zip file. Defaults to None.
            logger (Logger, optional): A custom logger instance. Defaults to None.
        """
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        super().__init__(root, url, zip_filename, redownload, reextract, max_fetch_retries, max_fetch_retry_delay,
                         logger)

    def _save_content(self):
        """
        Downloads the ZIP file from Google Drive using `gdown` and saves it to the zip path.

        Overrides the `_save_content` method in the parent `Downloader` class.

        Raises:
            gdown.exceptions.DownloadError: If the download fails.
        """
        gdown.download(str(self.url), str(self.zip_path), quiet=False)
