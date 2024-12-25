from typing import cast

import torch
from pytorch_lightning import LightningDataModule
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from illust_salmap.training.metrics import normalized


def train(model: Module, criterion: Module, dataloader: DataLoader, optimizer: Optimizer, device) -> dict:
    model.train()
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(dataloader, "test")):
        image, ground_truth = cast(tuple[Tensor, Tensor], batch)
        image, ground_truth = image.to(device), ground_truth.to(device)

        optimizer.zero_grad()
        predict = model(image)
        loss = criterion(predict, ground_truth)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return {"loss": epoch_loss / len(dataloader)}


@torch.no_grad()
def validation(model: Module, criterion: Module, dataloader: DataLoader, device) -> dict:
    model.eval()
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(dataloader, "validation")):
        image, ground_truth = cast(tuple[Tensor, Tensor], batch)
        image, ground_truth = image.to(device), ground_truth.to(device)
        predict = model(image)
        loss = criterion(predict, ground_truth)
        epoch_loss += loss.item()

    return {"loss": epoch_loss / len(dataloader)}


@torch.no_grad()
def visualize(title, model, dataloader, device):
    batch = next(iter(dataloader))
    images, ground_truths = cast(tuple[Tensor, Tensor], batch)
    images, ground_truths = images.to(device), ground_truths.to(device)

    predicts = model(images)

    # 画像を正規化
    images = normalized(images)
    ground_truths = normalized(ground_truths)
    predicts = normalized(predicts)

    # Matplotlibのプロットを作成
    fig, axes = plt.subplots(1, 3, figsize=(11, 8), dpi=350)

    # 入力画像
    axes[0].set_title('input image')
    axes[0].imshow(images[0].cpu().permute(1, 2, 0).detach().numpy())
    axes[0].axis("off")

    # グラウンドトゥルース画像
    axes[1].set_title('ground truth')
    axes[1].imshow(ground_truths[0].cpu().permute(1, 2, 0).detach().numpy())
    axes[1].axis("off")

    # 予測画像
    axes[2].set_title('predict')
    axes[2].imshow(predicts[0].cpu().permute(1, 2, 0).detach().numpy())
    axes[2].axis("off")

    # 画像をバッファに保存してTensorBoardに追加
    plt.tight_layout()
    plt.show()
    plt.close(fig)


@torch.no_grad()
def test(model: Module, criterion: Module, dataloader: DataLoader, device="cuda") -> dict:
    model.to(device)
    model.eval()
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, "test")):
        image, ground_truth = cast(tuple[Tensor, Tensor], batch)
        image, ground_truth = image.to(device), ground_truth.to(device)
        predict = model(image)
        loss = criterion(predict, ground_truth)
        epoch_loss += loss.item()

    return {"loss": epoch_loss / len(dataloader)}


def fit(model: Module, criterion: Module, datamodule: LightningDataModule, optimizer: Optimizer, epochs: int = 100,
        device="cuda"):
    model.to(device)

    datamodule.prepare_data()
    datamodule.setup("fit")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 100)

        train_metrics = train(model, criterion, datamodule.train_dataloader(), optimizer, device)
        val_metrics = validation(model, criterion, datamodule.val_dataloader(), device)
        visualize(epoch, model, datamodule.val_dataloader(), device)

        print(f"Training Loss: {train_metrics['loss']:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
