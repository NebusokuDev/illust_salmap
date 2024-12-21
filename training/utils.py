import random
from pathlib import Path

import numpy.random
import torch
from PIL import Image
from torch import cuda, backends
from torch.nn import Module
from tqdm import tqdm


def init_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False


def get_class_name(obj: object):
    return type(obj).__name__


def get_save_path(root: str | Path, datamodule, model: Module):
    model_name = get_class_name(model)
    module_name = get_class_name(datamodule)
    return Path(f"{root}/{module_name}/{model_name}")


def get_log_path(root, datamodule, model):
    return get_save_path(root, datamodule, model) / "logs"


def get_checkpoint_path(root, datamodule, model):
    return get_save_path(root, datamodule, model) / "checkpoints"


def calculate_mean_std(dataset, image=True, ground_truth=True):
    # 画像とサリエンシーマップの平均と標準偏差を格納する
    image_mean = torch.zeros(3, dtype=torch.float32)  # RGB画像の3チャネル分
    image_std = torch.zeros(3, dtype=torch.float32)
    map_mean = torch.zeros(1, dtype=torch.float32)  # サリエンシーマップは1チャネル
    map_std = torch.zeros(1, dtype=torch.float32)

    # 画像数とサリエンシーマップ数を取得
    num_images = len(dataset)

    # 画像とサリエンシーマップに対して計算
    for image_path, map_path in tqdm(dataset.image_map_pair_cache, desc="Calculating mean and std"):
        # 画像とサリエンシーマップの読み込み
        if image:
            image = Image.open(image_path).convert("RGB")
            image_tensor = dataset.image_transform(image)
            image_mean += image_tensor.mean(dim=(1, 2))  # 各チャネルの平均
            image_std += image_tensor.std(dim=(1, 2))  # 各チャネルの標準偏差

        if ground_truth:
            map_image = Image.open(map_path).convert("L")  # サリエンシーマップは1チャネル
            map_tensor = dataset.map_transform(map_image)
            map_mean += map_tensor.mean()
            map_std += map_tensor.std()

    # 最終的な平均と標準偏差を画像数で割る
    if image:
        image_mean /= num_images
        image_std /= num_images

    if ground_truth:
        map_mean /= num_images
        map_std /= num_images

    # 計算結果の表示
    if image:
        print("Image Mean:", image_mean)
        print("Image Std:", image_std)

    if ground_truth:
        print("Map Mean:", map_mean)
        print("Map Std:", map_std)

    return image_mean, image_std, map_mean, map_std
