import torch
from torch import Tensor
from torch.nn import (
    Module, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d,
    ConvTranspose2d, ReLU, LeakyReLU, Tanh, Sequential
)
from torchinfo import summary


class UNet(Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 3, activation: Module = LeakyReLU(), head: Module = Tanh()):
        super().__init__()
        self.encoder1 = EncoderBlock(in_channels, 64, activation=activation)
        self.encoder2 = EncoderBlock(64, 128, activation=activation)
        self.encoder3 = EncoderBlock(128, 256, activation=activation)
        self.encoder4 = EncoderBlock(256, 512, activation=activation)

        self.bottleneck = Bottleneck(512, 512, activation=activation)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, num_classes)
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottle = self.bottleneck(enc4)

        dec4 = self.decoder4(bottle, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        return self.head(dec1)


class EncoderBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.3, activation: Module = ReLU()):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 5, 1, 2)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)

        self.batch_norm1 = BatchNorm2d(out_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)

        self.max_pool = MaxPool2d(2, 2)

        self.dropout = Dropout2d(dropout_prob)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)

        x = self.max_pool(x)
        return x


class DecoderBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.2):
        super().__init__()
        self.conv_transpose = ConvTranspose2d(in_channels * 2, in_channels * 2, 4, 2, 1)
        self.conv = Conv2d(in_channels * 2, out_channels, 3, 1, 1)

        self.batch_norm1 = BatchNorm2d(in_channels * 2)
        self.batch_norm2 = BatchNorm2d(out_channels)

        self.dropout = Dropout2d(dropout_prob)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = torch.cat([x, y], dim=1)

        x = self.conv_transpose(x)
        x = self.batch_norm1(x)

        x = self.conv(x)
        x = self.batch_norm2(x)

        return self.dropout(x)


class Bottleneck(Module):
    def __init__(self, in_channels: int, out_channels: int, activation: Module = ReLU()):
        super().__init__()
        self.block = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            activation,
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


if __name__ == '__main__':
    model = UNet(1, 3)

    summary(model, (3, 384, 256))
