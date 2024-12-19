import random
from pathlib import Path

import numpy.random
import torch
from torch import cuda, backends
from torch.nn import Module

from models.dummy_net import DummyNet
from pytorch_lightning.callbacks import ModelCheckpoint


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


if __name__ == '__main__':
    print(get_save_path("./path", "to", DummyNet()))
