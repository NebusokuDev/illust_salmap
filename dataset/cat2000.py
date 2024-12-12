from pathlib import Path
from typing import Optional, Callable
from matplotlib import pyplot

from PIL import Image
from torch.utils.data import Dataset

from downloader.downloader import Downloader


class Cat2000Dataset(Dataset):
    URL = "http://saliency.mit.edu/trainSet.zip"

    def __init__(self,
                 root: str,
                 categories: Optional[list[str]] = None,
                 image_transform: Optional[Callable] = None,
                 map_transform: Optional[Callable] = None):
        self.categories = categories or ["*"]  # None の場合デフォルトで全カテゴリ
        self.image_transform = image_transform
        self.map_transform = map_transform
        self.downloader = Downloader(root=f"{root}/cat2000", url=self.URL)

        self.image_map_pair_cache = []
        self.downloader(on_complete=self.cache_image_map_paths)

    @property
    def dataset_path(self) -> Path:
        return self.downloader.extract_path / "Stimuli"

    def cache_image_map_paths(self):
        # カテゴリの展開（"*" を含む場合は全カテゴリ）
        if "*" in self.categories:
            expanded_categories = [p for p in self.dataset_path.iterdir() if p.is_dir()]
        else:
            expanded_categories = [self.dataset_path / category for category in self.categories]

        expanded_categories.sort()

        for category in expanded_categories:
            images_path_list = sorted(category.glob("*.jpg"))
            maps_path_list = sorted((category / "Output").glob("*.jpg"))
            self.image_map_pair_cache.extend(zip(images_path_list, maps_path_list))

    def __len__(self):
        return len(self.image_map_pair_cache)

    def __getitem__(self, index: int):
        image_path, map_path = self.image_map_pair_cache[index]

        image = Image.open(image_path).convert("RGB")
        map_image = Image.open(map_path).convert("L")

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.map_transform is not None:
            map_image = self.map_transform(map_image)

        return image, map_image


if __name__ == '__main__':
    dataset = Cat2000Dataset("./data")
    image, label = next(iter(dataset))

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].set_axis_off()

    axes[1].imshow(label)
    axes[1].set_title("label")
    axes[1].set_axis_off()
    fig.show()