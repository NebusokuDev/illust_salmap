import multiprocessing
from typing import Optional, Callable

from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Normalize, ToTensor, Compose

from calc_mean_std import calculate_mean_std
from downloader import GoogleDriveDownloader
from matplotlib import pyplot


class SALICONDataset(Dataset):
    IMAGE_ID = r"1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5"
    MAPS_ID = r"1PnO7szbdub1559LfjYHMy65EDC4VhJC8"

    def __init__(self,
                 root: str,
                 categories=None,
                 image_transform: Optional[Callable] = None,
                 map_transform: Optional[Callable] = None
                 ):

        self.categories = categories or ["test", "train"]

        self.image_transform = image_transform
        self.map_transform = map_transform

        self.image_downloader = GoogleDriveDownloader(f"{root}/salicon", self.IMAGE_ID, zip_filename="images.zip")
        self.map_downloader = GoogleDriveDownloader(f"{root}/salicon", self.MAPS_ID, zip_filename="maps.zip")

        self.image_downloader()
        self.map_downloader()

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []
        self.cache_image_map_paths()

    def cache_image_map_paths(self):
        for category in self.categories:
            images_dir = self.image_downloader.extract_path / category
            maps_dir = self.map_downloader.extract_path / category

            images_path_list = sorted(images_dir.glob("*.jpg"))
            maps_path_list = sorted(maps_dir.glob("*.png"))

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


class SALICONDataModule(LightningDataModule):
    def __init__(self, root: str = "./data", batch_size: int = 32, num_workers: int = multiprocessing.cpu_count()):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # データ変換
        self.image_transform = Compose([
            ToTensor(),
            Normalize(0.5, 0.5)
        ])

        self.map_transform = Compose([
            ToTensor(),
            Normalize(0.5, 0.5)
        ])

    def prepare_data(self):
        SALICONDataset(self.root)

    def setup(self, stage: str = None):
        salicon = SALICONDataset(self.root, map_transform=self.map_transform, image_transform=self.image_transform)
        total = len(salicon)

        n_train = int(total * 0.8)
        n_val = total - n_train

        (train, val) = random_split(dataset=salicon, lengths=[n_train, n_val])

        if stage == "fit" or stage is None:
            self.train = train
            self.val = val

        if stage == "test" or stage is None:
            self.test = val

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    dataset = SALICONDataset("./data")

    image, label = next(iter(dataset))

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].set_axis_off()

    axes[1].imshow(label)
    axes[1].set_title("label")
    axes[1].set_axis_off()
    fig.show()

    calculate_mean_std(SALICONDataset("./data", map_transform=ToTensor(), image_transform=ToTensor()))
