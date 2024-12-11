from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d, ConvTranspose2d, Sigmoid


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(out_channels),
            LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)

class Pix2PixDiscriminator(Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 1024),
        )

        self.patch_classifier = Sequential(
            ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            LeakyReLU(),
            BatchNorm2d(512),
            ConvTranspose2d(512, 1, kernel_size=4, stride=2, padding=1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.patch_classifier(x)
        return x