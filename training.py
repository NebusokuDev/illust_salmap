import torch
from torch.nn import MSELoss, Sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import Resize, ToTensor, Compose

from dataset import Cat2000Dataset
from models import UNetV2
from training import Trainer
from training.metrics import AreaUnderCurve

if __name__ == '__main__':
    image_transform = Compose([
        Resize((384, 256)),
        ToTensor()
    ])

    map_transform = Compose([
        Resize((384, 256)),
        ToTensor()
    ])

    model = UNetV2(head=Sigmoid())
    optimizer = Adam(model.parameters())

    dataset = Cat2000Dataset("./data", image_transform=image_transform, map_transform=map_transform)
    total = len(dataset)
    train_size = int(total * 0.8)
    test_size = total - train_size
    train, test = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train, batch_size=16)
    test_dataloader = DataLoader(test, batch_size=16)

    criterion = MSELoss(reduction='batchmean')

    metrics = {"AUC": AreaUnderCurve()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(train_dataloader, test_dataloader, criterion, device, metrics=metrics)
    trainer.fit(model, optimizer)
