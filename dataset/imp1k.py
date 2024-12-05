import glob
from os import path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset
from google.colab import drive

from google.colab import drive
import shutil


class Imp1kDataset(Dataset):
    URL = r"https://predimportance.mit.edu/data/imp1k.zip"


    def __init__(self,
                 root,
                 categories=None,
                 image_transform= None,
                 map_transform=None
                 ):

        self.categories = categories or ["ads", "infograpics", "movie_posters", "webpages"]

        self.image_transform = image_transform
        self.map_transform = map_transform

        print(f"url: {self.URL}")

        self.downloader = Downloader(root=root, url=self.URL)

        self.downloader()

        self.cache_image_map_paths_cashed = False

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []
        self.cache_image_map_paths()

    def cache_image_map_paths(self):
        if self.cache_image_map_paths_cashed:
            return

        for category in self.categories:
            images_dir = f"{self.downloader.extract_path}/imgs"
            maps_dir = f"{self.downloader.extract_path}/maps"

            images_path_list = sorted(glob.glob(path.join(images_dir, category, "*.jpg")))
            maps_path_list = sorted(glob.glob(path.join(maps_dir, category, "*.jpg")))

            # ペアリング
            for img_path, map_path in zip(images_path_list, maps_path_list):
                if path.basename(img_path) == path.basename(map_path):
                    self.image_map_pair_cache.append((img_path, map_path))

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

    def __str__(self):
        return "\n".join(
            f"image: {Image.open(pair[0]).size}, map: {Image.open(pair[1]).size}" for pair in self.image_map_pair_cache)