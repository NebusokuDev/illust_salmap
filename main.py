import argparse

parser = argparse.ArgumentParser(description="train model")

parser.add_argument("-e", "--epochs", type=int, required=False, default=50)
parser.add_argument("-b", "--batch-size", type=int, required=False, default=32)
parser.add_argument("-lr", type=float, required=False, default=0.0001)
parser.add_argument("-n", "--model-name", type=str, required=False, default="model")
parser.add_argument("-r", "--root", type=str, required=False, default="./")
parser.add_argument("-d", "--device", type=str, required=False, default="cpu")

args = parser.parse_args()
