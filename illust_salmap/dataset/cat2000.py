import multiprocessing
from typing import Optional, Callable

import torch
from PIL import Image
from matplotlib import pyplot
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.v2 import Resize, Compose, ToTensor, Normalize, Grayscale, ToDtype

from illust_salmap.training.utils import calculate_mean_std
from illust_salmap.downloader.downloader import Downloader


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

    def cache_image_map_paths(self):
        stimuli_path = self.downloader.extract_path / "Stimuli"
        fixation_path = self.downloader.extract_path / "FIXATIONMAPS"

        # カテゴリの展開（"*" を含む場合は全カテゴリ）
        if "*" in self.categories:
            expanded_categories = [p.name for p in stimuli_path.iterdir() if p.is_dir()]
        else:
            expanded_categories = [category for category in self.categories]

        expanded_categories.sort()

        for category in expanded_categories:
            stimuli_path_list = sorted((stimuli_path / category).glob("???.jpg"))
            fixation_path_list = sorted((fixation_path / category).glob("???.jpg"))

            self.image_map_pair_cache.extend(zip(stimuli_path_list, fixation_path_list))

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


class Cat2000(LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = multiprocessing.cpu_count(),
                 img_size=(256, 384)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # データ変換
        self.image_transform = Compose([
            Resize(img_size),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])

        self.map_transform = Compose([
            Resize(img_size),
            Grayscale(),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])

    def prepare_data(self):
        Cat2000Dataset(self.data_dir)

    def setup(self, stage: str = None):
        cat2000 = Cat2000Dataset(self.data_dir, map_transform=self.map_transform, image_transform=self.image_transform)
        total = len(cat2000)

        n_train = int(total * 0.8)
        n_val = total - n_train

        train, val = random_split(dataset=cat2000, lengths=[n_train, n_val])

        if stage == "fit" or stage is None:
            self.train = train
            self.val = val

        if stage == "test" or stage is None:
            self.test = val

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def __str__(self):
        return type(Cat2000Dataset).__name__


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

    dataset = Cat2000Dataset("./", image_transform=ToDtype(torch.float32), map_transform=ToTensor())
    calculate_mean_std(dataset)
