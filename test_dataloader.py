from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.utils import make_grid

from dataset import Cat2000Dataset
from matplotlib import pyplot


if __name__ == '__main__':
    transform = Compose([
        Resize((384, 256)),
        ToTensor(),
        Normalize(std=[0.5], mean=[0.5])
    ])
    dataset = Cat2000Dataset("./data", image_transform=transform, map_transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, pin_memory=True, num_workers=8, shuffle=True)

    image, label = next(iter(dataloader))

    print(image.shape)
    grid = make_grid(image, nrow=2)

    print(grid.shape)
