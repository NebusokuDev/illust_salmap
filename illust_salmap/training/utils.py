import io
import random
from pathlib import Path

import cv2
import numpy as np
import numpy.random
import torch
from PIL import Image

from numpy import ndarray
from torch import cuda, backends, Tensor
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


def get_log_path(root: str | Path, datamodule, model: Module):
    return get_save_path(root, datamodule, model) / "logs"


def get_checkpoint_path(root: str | Path, datamodule, model: Module):
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


def to_image(tensor: Tensor):
    if tensor.dim() != 3 or tensor.size(0) not in {1, 3}:
        raise ValueError(f"Invalid tensor shape: {tensor.shape}. Expected (C, H, W) with C=1 or 3.")
    return tensor.permute(1, 2, 0).detach().cpu().numpy()


def generate_plot(title: str, images: dict[str, Tensor], figsize=(11, 8), dpi=350):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(images.keys()), figsize=figsize, dpi=dpi)

    fig.suptitle(title)

    for ax, (name, image) in zip(axes, images.items()):
        ax.set_title(name)
        ax.set_axis_off()
        ax.imshow(to_image(image))

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer


def create_color_map(saliency_map: ndarray, colormap: int = cv2.COLORMAP_JET):
    assert saliency_map.dtype == np.uint8
    assert saliency_map.ndim == 2 or saliency_map.ndim == 3

    return cv2.applyColorMap(saliency_map, colormap)[:, :, ::-1]


def overlay_saliency_map(image: ndarray, saliency_map: ndarray, alpha: float = 0.5):
    assert saliency_map.dtype == np.uint8
    assert saliency_map.ndim == 2 or saliency_map.ndim == 3
    assert image.shape[:2] == saliency_map.shape[:2]  # 画像とサリエンシーマップのサイズが一致することを確認

    # サリエンシーマップのカラー化
    color_map = create_color_map(saliency_map)

    # オーバーレイのためのアルファブレンディング
    return cv2.addWeighted(image, 1 - alpha, color_map, alpha, 0)
