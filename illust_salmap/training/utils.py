from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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


def generate_plot(title: str, images: dict[str, Tensor], figsize=(11, 8), dpi=350):
    fig, axes = plt.subplots(1, len(images.keys()), figsize=figsize, dpi=dpi)

    fig.suptitle(title)

    for ax, (name, img) in zip(axes, images.items()):
        ax.set_title(name)
        ax.set_axis_off()
        ax.imshow(img.permute(1, 2, 0).detach().cpu().numpy())

    plt.tight_layout()

    with BytesIO() as buffer:
        plt.savefig(buffer, format="png")

        pil_image = Image.open(buffer).convert("RGB")

        buffer.close()

    plt.close(fig)
    return pil_image


def create_color_map(saliency_map: ndarray, colormap: int = cv2.COLORMAP_JET):
    assert saliency_map.dtype == np.uint8
    assert saliency_map.ndim == 2 or saliency_map.ndim == 3

    return cv2.applyColorMap(saliency_map, colormap)[:, :, ::-1]


def overlay_saliency_map(image: ndarray, saliency_map: ndarray, alpha: float = 0.5):
    assert saliency_map.dtype == np.uint8
    assert saliency_map.ndim == 2 or saliency_map.ndim == 3
    assert image.shape[:2] == saliency_map.shape[:2]

    color_map = create_color_map(saliency_map)

    return cv2.addWeighted(image, 1 - alpha, color_map, alpha, 0)


def clop_image_from_saliency_map(image, saliency_map, alpha=0.5):
    # サリエンシーマップの正規化（0〜1にスケール）
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    # 注目領域を抽出
    mask = saliency_map > alpha
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("注目領域が見つかりません。alphaの値を調整してください。")

    # クロップ領域を計算
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    crop_box = (*top_left[::-1], *bottom_right[::-1])  # (left, upper, right, lower)

    # 画像をクロップ
    cropped_image = image.crop(crop_box)
    return cropped_image


if __name__ == '__main__':
    writer = SummaryWriter("./tmp")

    for i in range(10):
        dummy_image = torch.rand(3, 32, 32)

        images = {
            "image": dummy_image, "saliency_map": dummy_image, "ground_truth": dummy_image,
        }

        image = generate_plot("test", images)
        writer.add_image("test", image, i)
        writer.flush()
