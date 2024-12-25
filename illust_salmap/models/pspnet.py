import torch
from torch.nn import Module, Conv2d, BatchNorm2d, Sequential, AvgPool2d, Upsample, ReLU, ModuleList, AdaptiveAvgPool2d
from torchinfo import summary
from torchvision.models import resnet50, ResNet50_Weights


class PSPNet(Module):
    def __init__(self):
        super().__init__()
        self.feature_map = FeatureMap()
        self.upsample = Upsample()
        self.pyramid_pool = PyramidPool()
        self.aux_loss = AUXLoss()
        self.decoder = Decoder()

    def forward(self, x):
        if self.training:
            aux = self.aux_loss(x)
            return x, aux

        return x


class FeatureMap(Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = Sequential(
            Conv2d(3, 64, 3, 2, 1),
            BatchNorm2d(64),
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

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
    def __init__(self, output_size, in_channels=2048, out_channels=512, size=(60, 60)):
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


class AUXLoss(Module):
    pass


class Decoder(Module):
    pass


class PSPNetLoss(Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, losses):
        loss, aux_loss = losses
        loss = self.criterion(loss)
        aux_loss = self.criterion(aux_loss)
        return loss + 0.4 * aux_loss


if __name__ == '__main__':
    summary(PSPNet(), (4, 3, 128, 128))
