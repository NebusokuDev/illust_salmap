import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
from dataset import Imp1kDataset
from models import UNet
from training.trainer import Trainer

if __name__ == '__main__':
    image_transform = Compose([ToTensor(), ])
    map_transform = Compose([ToTensor()])

    model = UNet()
    model.decoder_32_out.use_skip_connection = False

    optimizer = AdamW(model.parameters())

    imp1k = Imp1kDataset("./data", image_transform=image_transform, map_transform=map_transform)

    train_dataloader = DataLoader(imp1k)
    test_dataloader = DataLoader(imp1k)

    criterion = MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(train_dataloader, test_dataloader, criterion, device)
    trainer.fit(model, optimizer)
