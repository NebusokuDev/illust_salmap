from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor

from PIL import Image
from torch.utils.data import Dataset

from downloader.downloader import Downloader


class Imp1kCategories:
    ads = "ads"
    infographics = "infographics"
    movie_posters = "movie_posters"
    webpages = "webpages"
    all = [ads, infographics, movie_posters, webpages]


class Imp1kDataset(Dataset):
    URL = "https://predimportance.mit.edu/data/imp1k.zip"

    def __init__(self,
                 root,
                 categories=None,
                 image_transform=None,
                 map_transform=None
                 ):

        self.categories = categories or Imp1kCategories.all

        self.image_transform = image_transform
        self.map_transform = map_transform

        print(f"url: {self.URL}")

        self.downloader = Downloader(root=f"{root}/imp1k", url=self.URL)

        self.downloader()

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []
        self.cache_image_map_paths()

    def cache_image_map_paths(self):
        with ThreadPoolExecutor() as executor:
            futures: list[Future] = []
            for category in self.categories:
                # 画像とマップのパスを並列で処理
                futures.append(executor.submit(self._process_category, category))

            # すべてのタスクが終了するのを待つ
            for future in futures:
                future.result()

    def _process_category(self, category: str):
        images_dir = self.downloader.extract_path / category / "imgs"
        maps_dir = self.downloader.extract_path / category / "maps"

        images_path_list = sorted(images_dir.glob("*.jpg"))
        maps_path_list = sorted(maps_dir.glob("*.jpg"))

        # ペアリングしてキャッシュ
        self.image_map_pair_cache.extend(zip(images_path_list, maps_path_list))

    def __len__(self):
        return len(self.image_map_pair_cache)

    def __getitem__(self, index: int):
        image_path, map_path = self.image_map_pair_cache[index]

        image = Image.open(image_path)
        map_image = Image.open(map_path)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.map_transform is not None:
            map_image = self.map_transform(map_image)
        elif self.image_transform is not None:
            map_image = self.image_transform(map_image)

        return image, map_image
