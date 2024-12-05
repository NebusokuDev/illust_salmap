import argparse
import pathlib

import torchvision.datasets
from torch.nn import MSELoss, Conv2d
from torch.optim import Adam

from utils.utils import choose_device

parser = argparse.ArgumentParser(description="train model")

parser.add_argument("-e", "--epochs", type=int, required=False, default=50)
parser.add_argument("-b", "--batch-size", type=int, required=False, default=32)
parser.add_argument("-lr", type=float, required=False, default=0.0001)
parser.add_argument("-n", "--model-name", type=str, required=False, default="model")
parser.add_argument("-r", "--root", type=str, required=False, default="./")
parser.add_argument("-d", "--device", type=str, required=False, default="cpu")

args = parser.parse_args()

print(f"epochs: {args.epochs}")
print(f"batch size: {args.batch_size}")
print(f"lr: {args.lr}")
print(f"model name: {args.model_name}")
print(f"root: {pathlib.Path(args.root).resolve()}")

