from typing import Sequence

import torch
from torch import Tensor, SymInt
from torch.nn import *
from torch.nn.parameter import Parameter
from torchinfo import summary


class UNetV2(Module):
    def __init__(self, classes: int = 1, in_channels: int = 3, activation=LeakyReLU(), head=Tanh()):
        super().__init__()
        self.encoder1 = EncoderBlock(in_channels, 64, activation=activation)
        self.encoder2 = EncoderBlock(64, 128, activation=activation)
        self.encoder3 = EncoderBlock(128, 256, activation=activation)
        self.encoder4 = EncoderBlock(256, 512, activation=activation)

        self.bottleneck = BottleNeck(512, 512, activation=activation)

        self.decoder4 = DecoderBlock(512, 256, activation=activation)
        self.decoder3 = DecoderBlock(256, 128, activation=activation)
        self.decoder2 = DecoderBlock(128, 64, activation=activation)
        self.decoder1 = DecoderBlock(64, classes, activation=activation)
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

        self.se_block = SEBlock(out_channels)
        self.max_pool = MaxPool2d(2, 2)

        self.dropout = Dropout2d(dropout_prob)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.se_block(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


class DecoderBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.2, activation: Module = ReLU()):
        super().__init__()
        self.conv_transpose = ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.conv1 = Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)

        self.batch_norm1 = BatchNorm2d(out_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)

        self.dropout = Dropout2d(dropout_prob)
        self.skip_connector = SkipConnector()

        self.activation = activation

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        x = self.skip_connector(x, y)
        x = self.conv_transpose(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.batch_norm2(x)
        x = self.conv2(x)
        return self.dropout(x)


class BottleNeck(Module):
    def __init__(self, in_channels: int, out_channels: int, activation: Module = ReLU()):
        super().__init__()
        self.conv_block = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU()
        )

        self.global_attention = Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(out_channels, out_channels, kernel_size=1),
            Sigmoid()
        )

        self.activation = activation

    def forward(self, x):
        x = self.conv_block(x)
        attention_weights = self.global_attention(x)
        return self.activation(x * attention_weights)


class SEBlock(Module):
    def __init__(self, in_channels: int, reduction: int = 16, bias: bool = False):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Linear(in_channels, in_channels // reduction, bias=bias)
        self.relu = ReLU()
        self.fc2 = Linear(in_channels // reduction, in_channels, bias=bias)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor):
        squeeze = self.avg_pool(x).view(x.size(0), -1)
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(x.size(0), x.size(1), 1, 1)
        return x * excitation


class SkipConnector(Module):
    def __init__(self, skip_weight: float = 1, shape: Sequence[int | SymInt] = 1):
        super().__init__()
        self.skip_gate = Parameter(torch.ones(shape) * skip_weight)
        self.relu = ReLU()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if y is None:
            return x

        skip_connection = self.skip_gate * y
        skip_connection = torch.sigmoid(skip_connection)

        return self.relu(x + skip_connection)


if __name__ == '__main__':
    model = UNetV2(1, 3)

    summary(model, (4, 3, 256, 384))
