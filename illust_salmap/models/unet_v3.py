from typing import Sequence

import torch
from torch import Tensor, SymInt
from torch.nn import *
from torchinfo import summary

from illust_salmap.models.ez_bench import benchmark
from illust_salmap.training.saliency_model import SaliencyModel


class UNetV3(Module):
    def __init__(self, classes: int = 1,
                 in_channels: int = 3,
                 activation=SiLU(),
                 head=Sigmoid(),
                 num_blocks: int = 7,
                 base_channels: int = 32,
                 scale_stride=2,
                 ):
        super().__init__()

        mid_encoders = []
        mid_decoders = []

        for i in range(num_blocks):
            in_channel = base_channels * (2 ** i)
            out_channel = base_channels * (2 ** (i + 1))
            dilation = 2 ** (i + 1)
            downsample = i % scale_stride == 0
            mid_encoders.append(
                EncoderBlock(in_channel,
                             out_channel,
                             dilation,
                             activation=activation,
                             downsample=downsample))

        for i in range(num_blocks):
            in_channel = base_channels * (2 ** (i + 1))
            out_channel = base_channels * (2 ** i)
            upsample = i % scale_stride == 0
            mid_decoders.append(DecoderBlock(in_channel, out_channel, activation=activation, upsample=upsample))

        self.encoders = ModuleList([
            EncoderBlock(in_channels, base_channels, dilation=1, activation=activation),
            *mid_encoders
        ])

        bottleneck_channels = base_channels * (2 ** num_blocks)

        self.bottleneck = EncoderBlock(bottleneck_channels, bottleneck_channels, dilation=16, activation=activation,
                                       downsample=False)

        self.decoders = ModuleList([
            *reversed(mid_decoders),
            DecoderBlock(base_channels, classes, activation=activation),
        ])

        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)

        x = self.bottleneck(x)

        for decoder, encoder_output in zip(self.decoders, reversed(encoder_outputs)):
            x = decoder(x, encoder_output)

        return self.head(x)


class EncoderBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dilation: int = 1,
            dropout_prob: float = 0.1,
            activation: Module = SiLU(),
            downsample: bool = True,
    ):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, "same", dilation=dilation)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, "same")

        self.batch_norm1 = BatchNorm2d(out_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)

        self.se_block = SEBlock(out_channels)
        self.max_pool = Conv2d(out_channels, out_channels, 2, 2, bias=False)

        self.dropout = Dropout2d(dropout_prob)
        self.activation = activation
        self.shortcut = Conv2d(in_channels, out_channels, 1, 1, "same", bias=False)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.se_block(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)

        res = x + self.shortcut(identity)

        if self.downsample:
            res = self.max_pool(res)

        res = self.dropout(res)
        return res


class DecoderBlock(Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 dropout_prob: float = 0.2,
                 activation: Module = SiLU(),
                 upsample: bool = True,
                 ):
        super().__init__()
        self.upsample = upsample
        self.conv_transpose = ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)

        self.batch_norm1 = BatchNorm2d(out_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)

        self.dropout = Dropout2d(dropout_prob)
        self.skip_connector = SkipConnector()

        self.activation = activation

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        x = self.skip_connector(x, y)
        if self.upsample:
            x = self.conv_transpose(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.batch_norm2(x)

        return self.dropout(x)


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
    def __init__(self, skip_weight: float = 0.5, shape: Sequence[int | SymInt] = 1, activation: Module = ReLU()):
        super().__init__()
        self.skip_gate = Parameter(torch.ones(shape) * skip_weight)
        self.activation = activation

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if y is None:
            return x

        skip_connection = self.skip_gate * y
        skip_connection = torch.sigmoid(skip_connection)

        return self.activation(x + skip_connection)


def unet_v3(ckpt_path=None):
    model = SaliencyModel(UNetV3())
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)['state_dict']
        model.load_state_dict(state_dict)
        print("Successfully loaded the model")
    return model


if __name__ == '__main__':
    ckpt_path = input("ckpt path: ").strip("'").strip('"')
    model = unet_v3(ckpt_path=ckpt_path)
    shape = (4, 3, 256, 256)
    model(torch.randn(shape))
    summary(model, shape)
    benchmark(model, shape)
