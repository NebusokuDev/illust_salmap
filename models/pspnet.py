from torch.nn import Module
from torchvision.models import resnet50, ResNet50_Weights


class PSPNet(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FeatureMap(Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(ResNet50_Weights.IMAGENET1K_V2)

    def forward(self, x):
        pass


class PyramidPooling(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PoolingBlock(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Upsampler(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
