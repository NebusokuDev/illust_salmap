from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, BatchNorm2d, Flatten, Linear
from torchinfo import summary

from illust_salmap.models.ez_bench import benchmark


class SalGANDiscriminator(Module):
    # in_channels -> cat(image (3ch) + ground truth (1ch))
    # classes -> binary classification
    def __init__(self, in_channels=4, classes=1):
        super().__init__()

        self.features = Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        self.expansion = Flatten()

        self.classifier = Sequential(
            Linear(128 * 32 * 32, 256),
            ReLU(),
            Linear(256, 64),
            ReLU(),
            Linear(64, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.expansion(x)
        x = self.classifier(x)
        return x


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = Sequential(
            Conv2d(in_channels, out_channels, 3, 1, 1),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, 3, 1, 1),
            BatchNorm2d(out_channels),
            ReLU(),
            MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    model = SalGANDiscriminator()
    shape = (4, 4, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
