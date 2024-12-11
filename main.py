import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose
from torchvision.transforms.v2 import Resize
from torch.nn import L1Loss

from dataset import Imp1kDataset
from models import UNet
from training import Trainer

if __name__ == '__main__':
    image_transform = Compose([Resize((256, 256)), ToTensor()])
    map_transform = Compose([Resize((256, 256)), ToTensor()])

    model = UNet()
    model.decoder_32_out.use_skip_connection = False

    optimizer = Adam(model.parameters())

    imp1k = Imp1kDataset("./data", image_transform=image_transform, map_transform=map_transform)
    total = len(imp1k)
    train_size = int(total * 0.8)
    test_size = total - train_size
    train, test = random_split(imp1k, [train_size, test_size])

    train_dataloader = DataLoader(train, batch_size=16)
    test_dataloader = DataLoader(test, batch_size=16)

    criterion = MSELoss()

    metrics = {"mae": L1Loss()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(train_dataloader, test_dataloader, criterion, device, "imp1k_unet", metrics=metrics)
    trainer.fit(model, optimizer)
