import multiprocessing
from torchvision import transforms
from PIL import Image

from matplotlib import pyplot
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from calc_mean_std import calculate_mean_std
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

        for category in self.categories:
            images_dir = self.downloader.extract_path / "imgs" / category
            maps_dir = self.downloader.extract_path / "maps" / category
            images_path_list = sorted(images_dir.glob("*"))
            maps_path_list = sorted(maps_dir.glob("*"))

            self.image_map_pair_cache.extend((zip(images_path_list, maps_path_list)))

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


class Imp1k(LightningDataModule):
    def __init__(self, root: str = "./data", batch_size: int = 64, num_workers: int = multiprocessing.cpu_count(),
                 img_size=256):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # データ変換
        self.image_transform = Compose([
            Resize(img_size),
            PadToSquare(),
            ToTensor(),
            Normalize(0.5, 0.5)
        ])

        self.map_transform = Compose([
            Resize(img_size),
            PadToSquare(),
            ToTensor(),
            Normalize(0.5, 0.5)
        ])

    def prepare_data(self):
        Imp1kDataset(self.root)

    def setup(self, stage: str = None):
        imp1k = Imp1kDataset(self.root, map_transform=self.map_transform, image_transform=self.image_transform)
        total = len(imp1k)

        n_train = int(total * 0.8)
        n_val = total - n_train

        (train, val) = random_split(dataset=imp1k, lengths=[n_train, n_val])

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


class PadToSquare(object):
    def __init__(self, fill=0):
        self.fill = fill  # パディングの色（デフォルトは黒）

    def __call__(self, image):
        # 画像のサイズを取得
        width, height = image.size

        # 最長辺の長さを取得して正方形にするための新しいサイズを決定
        size = max(width, height)

        # 左右と上下に追加するパディングの量を計算
        padding_left = (size - width) // 2
        padding_top = (size - height) // 2
        padding_right = size - width - padding_left
        padding_bottom = size - height - padding_top

        # パディングを加えて正方形にする
        return transforms.functional.pad(image, (padding_left, padding_top, padding_right, padding_bottom),
                                         fill=self.fill)


if __name__ == '__main__':
    pad_and_resize = Compose([
        Resize(256),
        PadToSquare()
    ])

    dataset = Imp1kDataset("./data", image_transform=pad_and_resize, map_transform=pad_and_resize)
    image, label = next(iter(dataset))

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].set_axis_off()

    axes[1].imshow(label)
    axes[1].set_title("label")
    axes[1].set_axis_off()
    fig.show()

    dataset = Imp1kDataset("./data", image_transform=ToTensor(), map_transform=ToTensor())
    calculate_mean_std(dataset)
