import random
from abc import ABC, abstractmethod
from typing import Tuple, Literal, Optional, List

import numpy as np
import torch
from PIL.Image import Image
from numpy import ndarray
from torch import Tensor
from torchvision.transforms.v2 import functional as F


class Transform(ABC):
    @abstractmethod
    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        pass

    def __call__(self, *data):
        return self.convert(*data)


class ToTensor(Transform):
    def __init__(self, dtype: Optional[List[torch.dtype]] = None):
        # dtypeが指定されていない場合はtorch.float32をデフォルトとして設定
        if dtype is None:
            dtype = [torch.float32]
        self.dtypes = dtype

    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []

        # dtypesが1つの場合、すべてのデータに同じdtypeを適用
        if len(self.dtypes) == 1:
            dtypes = [self.dtypes[0]] * len(data)
        # dtypesとデータの長さが一致する場合
        elif len(self.dtypes) == len(data):
            dtypes = self.dtypes
        else:
            raise ValueError("Number of dtypes does not match the number of data inputs.")

        # データに対して指定されたdtypeを適用
        for item, dtype in zip(data, dtypes):
            if isinstance(item, Image):
                result.append(torch.tensor(np.array(item), dtype=dtype))
            elif isinstance(item, ndarray):
                result.append(torch.tensor(item, dtype=dtype))
            elif isinstance(item, Tensor):
                result.append(item.to(dtype))

        return tuple(result)


class ToNDArray(Transform):
    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []
        for item in data:
            if isinstance(item, Tensor):
                result.append(item.numpy())
            elif isinstance(item, Image):
                result.append(np.array(item))
            elif isinstance(item, ndarray):
                result.append(item)
        return tuple(result)


class ToPIL(Transform):
    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []
        for item in data:
            if isinstance(item, Tensor):
                item = item.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC
                result.append(Image.fromarray(item.astype(np.uint8)))
            elif isinstance(item, ndarray):
                result.append(Image.fromarray(item.astype(np.uint8)))
            elif isinstance(item, Image):
                result.append(item)
        return tuple(result)

class Resize(Transform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (height, width)

    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []
        for item in data:
            if isinstance(item, Tensor):
                item = F.resize_image

                # Tensorの場合、PIL Imageに変換してからリサイズ
                item = item.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                item = np.array(Image.fromarray(item.astype(np.uint8)).resize(self.size))
                item = torch.tensor(item).permute(2, 0, 1)  # HWC -> CHW
                result.append(item)
            elif isinstance(item, ndarray):
                # ndarrayの場合、PIL Imageに変換してからリサイズ
                item = np.array(Image.fromarray(item.astype(np.uint8)).resize(self.size))
                result.append(item)
            elif isinstance(item, Image):
                # PIL Imageの場合、直接リサイズ
                result.append(item.resize(self.size))

        return tuple(result)


class RandomFlip(Transform):
    def __init__(self, flip_type: Literal["horizontal", "vertical"] = "horizontal"):
        self.flip_type = flip_type  # "horizontal" または "vertical"

    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []

        is_flip = random.randint(0, 1) == 1

        if not is_flip:
            return data

        for item in data:
            if isinstance(item, Tensor):
                if self.flip_type == "horizontal":
                    item = F.hflip(item)
                    item = torch.flip(item, dims=[-1])  # 横方向に反転 (幅方向)
                elif self.flip_type == "vertical":
                    item = torch.flip(item, dims=[-2])  # 縦方向に反転 (高さ方向)
                result.append(item)
            elif isinstance(item, ndarray):
                if self.flip_type == "horizontal":
                    item = np.flip(item, axis=1)  # 横方向に反転 (幅方向)
                elif self.flip_type == "vertical":
                    item = np.flip(item, axis=0)  # 縦方向に反転 (高さ方向)
                result.append(item)
            elif isinstance(item, Image):
                if self.flip_type == "horizontal":
                    result.append(item.transpose(Image.FLIP_LEFT_RIGHT))  # 横方向に反転
                elif self.flip_type == "vertical":
                    result.append(item.transpose(Image.FLIP_TOP_BOTTOM))  # 縦方向に反転

        return tuple(result)


class RandomRotate(Transform):
    def __init__(self, max_angle: float = 30.0):
        self.max_angle = max_angle

    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []
        for item in data:
            angle = random.uniform(-self.max_angle, self.max_angle)  # ランダムに回転角度を決定

            if isinstance(item, Tensor):
                item = item.permute(1, 2, 0).cpu().numpy()
                item = np.array(Image.fromarray(item.astype(np.uint8)).rotate(angle))
                item = torch.tensor(item).permute(2, 0, 1)  # 再度CHW形式に戻す
                result.append(item)
            elif isinstance(item, ndarray):
                item = np.array(Image.fromarray(item.astype(np.uint8)).rotate(angle))
                result.append(item)
            elif isinstance(item, Image):
                result.append(item.rotate(angle))

        return tuple(result)


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []
        for item in data:
            if isinstance(item, Tensor):
                result.append(F.normalize(item, mean=self.mean, std=self.std))
            elif isinstance(item, ndarray):
                item = (item - self.mean) / self.std
                result.append(item)
            elif isinstance(item, Image):
                raise ValueError("PIL images cannot be normalized with this transform.")
        return tuple(result)


class Normalize01(Transform):
    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []
        for item in data:
            if isinstance(item, Tensor):
                result.append((item - item.min()) / (item.max() - item.min()))
            elif isinstance(item, ndarray):
                result.append((item - item.min()) / (item.max() - item.min()))
            elif isinstance(item, Image):
                np_img = np.array(item)
                np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min())
                result.append(Image.fromarray(np_img.astype(np.uint8)))
        return tuple(result)


class Lambda(Transform):
    def __init__(self, func):
        self.func = func

    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        result = []
        for item in data:
            if isinstance(item, (Tensor, ndarray, Image)):  # データの型チェック
                result.append(self.func(item))
            else:
                raise TypeError(f"Unsupported data type: {type(item)}")
        return tuple(result)


class Sequential(Transform):
    def __init__(self, *transforms: Transform):
        self.transforms = transforms

    def convert(self, *data: Tuple[Tensor | Image | ndarray]):
        for transform in self.transforms:
            data = transform.convert(*data)
        return data
