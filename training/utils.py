import random
from pathlib import Path

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


def get_save_path(root, dataset, model):
    return Path(f"{root}/{dataset}/{model}")


def get_log_path(root, dataset, model):
    return get_save_path(root, dataset, model) / "logs"


def get_checkpoint_path(root, dataset, model):
    return get_save_path(root, dataset, model) / "checkpoint"
