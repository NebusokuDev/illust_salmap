import torch
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, Tanh, Dropout2d, BatchNorm2d, MaxPool2d

from models import ImageShapeAdjuster


class EncoderBlock(Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.block = Sequential(
            Conv2d(in_channels, out_channels, 3, 1, 1),
            MaxPool2d(2, 2),
            BatchNorm2d(out_channels),
            LeakyReLU(),
            Dropout2d(dropout_prob),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_prob=.2):
        super().__init__()
        self.block = Sequential(
            ConvTranspose2d(in_channels + skip_channels, out_channels, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(out_channels),
            LeakyReLU(),
            Dropout2d(dropout_prob)
        )

    def forward(self, x, y=None):
        if y is None:
            return self.block(x)
        return self.block(torch.cat((x, y), dim=1))


class Pix2PixGenerator(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.image_shape_adjuster = ImageShapeAdjuster()

        self.encoder1 = EncoderBlock(in_channels, 32)
        self.encoder2 = EncoderBlock(32, 64)
        self.encoder3 = EncoderBlock(64, 128)
        self.encoder4 = EncoderBlock(128, 256)
        self.encoder5 = EncoderBlock(256, 512)
        self.encoder6 = EncoderBlock(512, 1024)

        self.bottleneck = Conv2d(1024, 1024, kernel_size=3, padding="same")

        self.decoder6 = DecoderBlock(1024, 1024, 512)
        self.decoder5 = DecoderBlock(512, 512, 256)
        self.decoder4 = DecoderBlock(256, 256, 128)
        self.decoder3 = DecoderBlock(128, 128, 64)
        self.decoder2 = DecoderBlock(64, 64, 32)
        self.decoder1 = DecoderBlock(32, 32, out_channels)
        self.output = Tanh()

    def forward(self, x):
        x = self.image_shape_adjuster.pad(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)

        bottle = self.bottleneck(e6)

        d6 = self.decoder6(bottle, e6)
        d5 = self.decoder5(d6, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        output = self.output(d1)

        return self.image_shape_adjuster.crop(output)

