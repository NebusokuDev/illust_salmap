import torch
from torch.nn import Module, Conv2d, MaxPool2d, ReLU, BatchNorm2d, Sequential, Linear, Dropout
from torchsummary import summary


class VGG16(Module):
    def __init__(self, classes, in_channels=3):
        super().__init__()
        self.features = Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
        )

        self.classifier = Sequential(
            Linear(512 * 8 * 8, 4096),
            ReLU(),
            Linear(4096, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x)
        x = self.classifier(x)

        return x


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)
        self.batch_norm1 = BatchNorm2d(out_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)
        self.activation = ReLU()
        self.max_pool = MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.max_pool(x)

        return x


if __name__ == '__main__':
    model = VGG16(classes=10)

    summary(model, (3, 256, 256))
