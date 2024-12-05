import glob
import os
from os import path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset

from downloader.downloader import Downloader


class Cat2000Dataset(Dataset):

    URL = "http://saliency.mit.edu/trainSet.zip"

    def __init__(self,
                 root: str = "/content/data",
                 categories: Optional[list[str]] = None,
                 image_transform: Optional[Callable] = None,
                 map_transform: Optional[Callable] = None,
                 ):

        if categories is None:
            categories = ["*"]

        self.categories = categories
        self.image_transform = image_transform
        self.map_transform = map_transform
        self.downloader = Downloader(root=f"{root}/cat2000", url=self.URL)

        self.image_map_pair_cache = []
        self.cache_image_map_paths_cashed = False
        self.downloader(on_complete=self.cache_image_map_paths)


    def cache_image_map_paths(self):
        if self.cache_image_map_paths_cashed:
            return

        dataset_path = path.join(self.downloader.extract_path, "Stimuli")

        # categoriesにワイルドカードが含まれている場合、全カテゴリディレクトリを展開
        if "*" in self.categories:
            expanded_categories = [d for d in glob.glob(os.path.join(dataset_path, "*")) if os.path.isdir(d)]
        else:
            expanded_categories = [os.path.join(dataset_path, category) for category in self.categories]

        expanded_categories.sort()

        for category in expanded_categories:
            category_path = path.join(dataset_path, category)
            images_path_list = glob.glob(path.join(category_path, "*.jpg"))
            maps_path_list = glob.glob(path.join(category_path, "Output", "*.jpg"))

            images_path_list.sort()
            maps_path_list.sort()

            # ペアリング
            for img_path, map_path in zip(images_path_list, maps_path_list):
                self.image_map_pair_cache.append((img_path, map_path))

    def __len__(self):
        return len(self.image_map_pair_cache)

    def __getitem__(self, idx: int):
        image_path, map_path = self.image_map_pair_cache[idx]
        image = Image.open(image_path).convert("RGB")
        map_image = Image.open(map_path).convert("RGB")



        if self.image_transform is not None:
            image = self.image_transform(image)

            if self.map_transform is not None:
                map_image = self.map_transform(map_image)
            else:
                map_image = self.image_transform(map_image)

        return image, map_image

    def __str__(self):
        return "\n".join(
            f"image: {Image.open(pair[0]).size}, map: {Image.open(pair[1]).size}" for pair in self.image_map_pair_cache)
