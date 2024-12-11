import gdown

from downloader import Downloader


class GoogleDriveDownloader(Downloader):
    def __init__(self,
                 root: str,
                 file_id: str,
                 redownload: bool = False,
                 reextract: bool = False,
                 zip_filename: str = None,
                 ):

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        super().__init__(root, url, zip_filename, redownload, reextract)

    def _save_content(self):
        gdown.download(str(self.url), str(self.zip_path), quiet=False)