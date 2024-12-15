import random

import numpy.random
import torch
from torch import cuda, backends


def init_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False


def normalize01(image):
    min_val = torch.min(image)
    max_val = torch.max(image)

    # 分母が 0 になる場合の処理
    if max_val == min_val:
        return torch.zeros_like(image)

    return (image - min_val) / (max_val - min_val)
