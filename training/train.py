import torch
from matplotlib import pyplot
from torch import Tensor
from torch.nn import Module, MSELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dataset.cat2000 import Cat2000
from models.unet_v2 import UNetV2


def train(model: Module, train_dataloader: DataLoader, criterion: Module, optimizer: Optimizer, device: torch.device):
    for batch_idx, (image, ground_truth) in enumerate(train_dataloader):
        optimizer.zero_grad()

        predict = model(image)
        loss: Tensor = criterion(predict)

        loss.backward()
        optimizer.step()


def validate(model: Module, val_dataloader: DataLoader, criterion: Module, device):
    for batch_idx, (image, ground_truth) in enumerate(val_dataloader):
        image = image.to(device)
        ground_truth = ground_truth.to(device)

        predict = model(image)

        loss = criterion(predict, ground_truth)
        loss.backward()

        print(f"{loss.item()}")


def fit(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs=50):
    for epoch in range(epochs):
        train(model, train_dataloader, criterion, optimizer, device)

        with torch.no_grad():
            validate(model, val_dataloader, criterion, device)


def test(model: Module, test_dataloader: DataLoader, criterion: Module, device):
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (image, ground_truth) in enumerate(test_dataloader):
            image = image.to(device)
            ground_truth = ground_truth.to(device)

            predict = model(image)

            loss = criterion(predict, ground_truth)
            total_loss += loss.item()

            print(f"{total_loss / (batch_idx + 1):.4f}")

            fig, axes = pyplot.subplots(ncols=3, )
            axes[0].imshow(image[0].permute(1, 2, 0))
            axes[1].imshow(ground_truth[0].permute(1, 2, 0))
            axes[2].imshow(predict[0].permute(1, 2, 0))
            pyplot.show()


if __name__ == '__main__':
    model = UNetV2()
    cat2000 = Cat2000("../data")
    cat2000.setup()
    dataloader = cat2000.test_dataloader()

    test(model, dataloader, MSELoss(), "cpu")
