import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from zipfile import BadZipFile
import requests
from tqdm import tqdm

KIB = 2 ** 10


class Downloader:
    def __init__(self,
                 root: str,
                 url: str,
                 zip_filename: str = None,
                 redownload: bool = False,
                 reextract: bool = False,
                 max_retries: int = 3,
                 retry_delay: int = 2,
                 ):

        self._root = Path(root).resolve()
        self.url = url
        self.zip_filename = zip_filename or Path(urlparse(self.url).path).name
        self.zip_path = self._root / self.zip_filename
        self.extract_path = self.zip_path.with_suffix("")

        self.reextract = reextract
        self.redownload = redownload

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def is_downloaded(self):
        return self.zip_path.exists()

    @property
    def is_extracted(self):
        return self.extract_path.exists()

    def download(self):
        """データセットをダウンロードする"""
        print(f"Downloading from {self.url}...")

        self._root.mkdir(parents=True, exist_ok=True)

        for attempt in range(self.max_retries):
            try:
                self._save_content()
                print("Data downloaded successfully.")
                break
            except requests.RequestException as err:
                print(f"Attempt {attempt}: エラーが発生しました - {err}")
                if hasattr(err, 'response') and err.response is not None:
                    if err.response.status_code != 200:
                        print(f"無効なステータスコード: {err.response.status_code}")
                        break

                print("再試行します...")
                time.sleep(self.retry_delay)
        else:
            print("リトライ回数の上限に達しました。")

    def _save_content(self):
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        with self.zip_path.open("wb") as f:
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1000) as progress_bar:
                for chunk in response.iter_content(chunk_size=100 * KIB):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

    def extract(self):
        print(f"Unzipping {self.zip_path}...")

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            top_level_dirs = {Path(x).parts[0] for x in zip_ref.namelist()}

            if len(top_level_dirs) == 1:
                top_level_dir = next(iter(top_level_dirs))
                self.extract_path = self._root / top_level_dir
                print(f"Extracting into {self.extract_path}...")

            total_files = len(zip_ref.namelist())
            with tqdm(total=total_files, unit='file') as progress:
                for file in zip_ref.namelist():
                    destination = self._root if len(top_level_dirs) == 1 else self.extract_path
                    zip_ref.extract(file, destination)
                    progress.update(1)

        print(f"Extracted to {self.extract_path}.")

    def __call__(self, on_complete=None):
        if self.redownload or not self.is_downloaded:
            print("Downloading...")
            self.download()
        else:
            print(f"Dataset already exists at {self.zip_path}, skipping download.")

        if self.reextract or not self.is_extracted:
            print("Extracting...")
            for retry in range(self.max_retries):
                try:
                    self.extract()
                except BadZipFile:
                    print("BadZipFile detected. Redownloading the file...")
                    self.zip_path.unlink(missing_ok=True)
        else:
            print(f"Dataset already extracted at {self.extract_path}, skipping extraction.")

        if on_complete is not None:
            on_complete()
