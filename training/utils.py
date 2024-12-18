import random
from pathlib import Path

import numpy.random
import torch
from pytorch_lightning import Trainer
from torch import cuda, backends
from torch.nn import Module

from models.dummy_net import DummyNet
from models.saliency_model import SaliencyModel


def init_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False

def get_class_name(obj: object):
    return type(obj).__name__


def get_save_path(root: str | Path, dataset, model: Module):
    model_name = get_class_name(model)
    return Path(f"{root}/{dataset}/{model_name}")


def get_log_path(root, dataset, model):
    return get_save_path(root, dataset, model) / "logs"


def get_checkpoint_path(root, dataset, model):
    return get_save_path(root, dataset, model) / "checkpoint"

if __name__ == '__main__':
    print(get_save_path("./path", "to", DummyNet()))