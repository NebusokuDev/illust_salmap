import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models


# ResNetバックボーン
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Pyramid Pooling Module
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_size = x.size()[2:]
        pooled_outputs = [F.interpolate(pool(x), size=input_size, mode='bilinear', align_corners=False) for pool in
                          self.pools]
        x = torch.cat([x] + pooled_outputs, dim=1)
        return self.relu(self.conv(x))


# PSPNet
class PSPNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.pyramid_pooling = PyramidPooling(2048, pool_sizes=[1, 2, 3, 6])
        self.final_conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        x = self.backbone(x)
        x = self.pyramid_pooling(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


if __name__ == '__main__':
    model = PSPNet()
    summary(PSPNet(), (3, 256, 256))

    output = model(torch.randn(1, 3, 256, 256))
    print(output.shape)
