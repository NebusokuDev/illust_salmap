import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv1(x)
        x2 = self.conv3_1(x)
        x3 = self.conv3_2(x)
        x4 = self.conv3_3(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.out_conv(x)
        return x


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(in_channels, 256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.classifier(x)
        return x


class Resnet50Backbone(nn.Module):
    def __init__(self):
        super(Resnet50Backbone, self).__init__()
        resnet = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1, backbone=Resnet50Backbone()):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.head = DeepLabHead(2048, num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.backbone(x)
        x = self.head(features)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


# 使用例
if __name__ == "__main__":
    from torchvision.models import resnet50

    model = DeepLabV3()

    # テスト入力
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(output.shape)
