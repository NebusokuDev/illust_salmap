import torch
from torch import nn
from torch.nn import Module
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.pooling = PyramidPooling()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled_features = self.pooling(features)
        output = self.classifier(pooled_features)
        return output


class Classifier(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(2048 + 512 * 4, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.classifier(x)


class FeatureMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class PyramidPooling(nn.Module):
    def __init__(self, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.branches = nn.ModuleList([
            FeatureMap(2048, 512) for _ in pool_sizes
        ])

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pooled = [x]
        for branch, size in zip(self.branches, self.pool_sizes):
            pooled_x = nn.functional.adaptive_avg_pool2d(x, output_size=size)
            upsampled_x = nn.functional.interpolate(branch(pooled_x), size=(h, w), mode='bilinear', align_corners=False)
            pooled.append(upsampled_x)
        return torch.cat(pooled, dim=1)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    summary(PSPNet(10), (3, 256, 256))
