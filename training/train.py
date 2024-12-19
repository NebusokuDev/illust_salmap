import torch
from torch.utils.data import DataLoader


def train(model, dataloader, criterion, optimizer, device, batch_stride=3):
    model.train()
    for batch_idx, (image, salmap) in enumerate(dataloader):
        image = image.to(device)
        salmap = salmap.to(device)

        optimizer.zero_grad()

        outputs = model(image)
        loss = criterion(outputs, salmap)
        loss.backward()
        optimizer.step()
        if batch_idx % batch_stride == 0:
            print(f"batch: {batch_idx:>5}/{len(dataloader):<5} ({batch_idx / len(dataloader):5.2%})",
                  f"loss: {loss.item():>9.4f}",
                  sep="\t")


def validate(model, dataloader, criterion, device, batch_stride=2):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (image, salmap) in enumerate(dataloader):
            image = image.to(device)
            salmap = salmap.to(device)

            outputs = model(image)
            loss = criterion(outputs, salmap)
            total_loss += loss.item()

            if batch_idx % batch_stride == 0:
                avg_loss = total_loss / (batch_idx + 1)

                print(f"batch: {batch_idx:>5}/{len(dataloader):<5} ({batch_idx / len(dataloader):>5.2%})",
                      f"loss: {loss:9.4f}",
                      f"avg loss: {avg_loss:9.4f}",
                      sep="\t")


def fit(model, train_dataloader: DataLoader, validate_dataloader: DataLoader, criterion, optimizer, epochs, device):
    for epoch in range(1, epochs + 1):
        train(model, train_dataloader, criterion, optimizer, device)
        validate(model, validate_dataloader, criterion, device)