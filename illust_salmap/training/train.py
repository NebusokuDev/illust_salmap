import torch
from pytorch_lightning import LightningDataModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model: Module, criterion: Module, dataloader: DataLoader, optimizer: Optimizer) -> dict:
    model.train()
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        image, ground_truth = batch

        optimizer.zero_grad()
        predict = model(image)
        loss = criterion(predict, ground_truth)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    return {"loss": epoch_loss / len(dataloader)}


def validation(model: Module, criterion: Module, dataloader: DataLoader) -> dict:
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            image, ground_truth = batch
            predict = model(image)
            loss = criterion(predict, ground_truth)
            epoch_loss += loss.item()

    return {"loss": epoch_loss / len(dataloader)}


def test(model: Module, criterion: Module, dataloader: DataLoader) -> dict:
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            image, ground_truth = batch
            predict = model(image)
            loss = criterion(predict, ground_truth)
            epoch_loss += loss.item()

    return {"loss": epoch_loss / len(dataloader)}


def fit(model: Module, criterion: Module, datamodule: LightningDataModule, optimizer: Optimizer, epochs: int = 100):
    datamodule.prepare_data()
    datamodule.setup("fit")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 100)

        train_metrics = train(model, criterion, datamodule.train_dataloader(), optimizer)
        val_metrics = validation(model, criterion, datamodule.val_dataloader())

        print(f"Training Loss: {train_metrics['loss']:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
