from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset

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
            else:
                map_image = self.image_transform(map_image)

        return image, map_image


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