import torch
from torch.nn import (AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout2d, MaxPool2d, Module, ReLU, Sequential, Upsample,
                      ConvTranspose2d)
from torch.nn.functional import interpolate
from torchinfo import summary
from torchvision.models import ResNet50_Weights, resnet50

from illust_salmap.models.ez_bench import benchmark
from illust_salmap.training.saliency_model import SaliencyModel


class PSPNet(Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.feature_map = FeatureMap(in_channels=in_channels)
        self.pyramid_pool = PyramidPool()
        self.aux_loss = AUXLoss()
        self.upscaler = Upscaler(2048, num_classes)

    def forward(self, x):
        size = x.shape[-2:]
        x, tmp = self.feature_map(x)
        x = self.pyramid_pool(x)
        x = self.upscaler(x, size)

        if self.training:
            aux = self.aux_loss(tmp, size=size)
            return x, aux

        return x


class FeatureMap(Module):
    def __init__(self, in_channels=3):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.layer0 = Sequential(Conv2d(in_channels, 64, 3, 2, 1),
                                 BatchNorm2d(64),
                                 ReLU(),
                                 Conv2d(64, 64, 3, 1, 1),
                                 BatchNorm2d(64),
                                 ReLU(),
                                 Conv2d(64, 128, 3, 1, 1),
                                 BatchNorm2d(128),
                                 ReLU(),
                                 MaxPool2d(3, 2, 1))

        self.layer1 = backbone.layer1
        self.layer1[0].conv1 = Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, bias=False)
        self.layer1[0].downsample[0] = Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=1, bias=False)
        self.layer2 = backbone.layer2
        self.layer3 = self.mod_conv(backbone.layer3, dilation=(2, 2), padding=(2, 2), stride=(1, 1))
        self.layer4 = self.mod_conv(backbone.layer4, dilation=(4, 4), padding=(4, 4), stride=(1, 1))

    def mod_conv(self, layer: Module, *, dilation: tuple[int, int], padding: tuple[int, int], stride: tuple[int, int]):
        for name, module in layer.named_modules():
            if "conv2" in name:
                module.dilation, module.padding, module.stride = dilation, padding, stride
            elif "downsample.0" in name:
                module.stride = stride

        return layer

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        aux = x
        x = self.layer4(x)

        return x, aux


class PyramidPool(Module):
    def __init__(self):
        super().__init__()

        self.pool1x1 = PyramidPoolBlock(output_size=1)
        self.pool2x2 = PyramidPoolBlock(output_size=2)
        self.pool3x3 = PyramidPoolBlock(output_size=3)
        self.pool6x6 = PyramidPoolBlock(output_size=6)

    def forward(self, x):
        p1x1 = self.pool1x1(x)
        p2x2 = self.pool2x2(x)
        p3x3 = self.pool3x3(x)
        p6x6 = self.pool6x6(x)

        return torch.cat([p1x1, p2x2, p3x3, p6x6], dim=1)


class PyramidPoolBlock(Module):
    def __init__(self, output_size, in_channels=2048, out_channels=512, size=(64, 64)):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(output_size)
        self.conv = Conv2d(in_channels, out_channels, 1)
        self.bn = BatchNorm2d(out_channels)
        self.activation = ReLU()
        self.upsample = Upsample(size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x


class Upscaler(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x, size=(256, 256)):
        x = interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.conv(x)

        return x


class DecoderBlock(Module):
    def __init__(self, in_channels, out_channels, activation=ReLU()):
        super().__init__()
        self.conv_transpose = ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)
        self.batch_norm1 = BatchNorm2d(in_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x


class AUXLoss(Module):
    def __init__(self, in_channels=1024, num_classes=1):
        super().__init__()
        self.conv = Conv2d(in_channels, 256, 3, 1, bias=False)
        self.bn = BatchNorm2d(256)
        self.activation = ReLU()
        self.dropout = Dropout2d(0.1)
        self.conv2 = Conv2d(256, num_classes, 1)

    def forward(self, x, size):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return interpolate(x, size=size, mode='bilinear', align_corners=False)


def unet_v2(ckpt_path=None):
    model = SaliencyModel(PSPNet())
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)['state_dict']
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    model = PSPNet()
    shape = (4, 3, 512, 512)
    summary(model, shape)
    benchmark(model, shape)
