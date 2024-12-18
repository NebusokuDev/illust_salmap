from torch.nn import Module
from torchvision.models import resnet50, ResNet50_Weights



class PSPNet(Module):
    def __init__(self):
        super().__init__()
        self.feature_map = FeatureMap()
        self.pyramid_pool = PyramidPooling()
        self.upsampler = Upsampler()

    def forward(self, x):
        return x


class FeatureMap(Module):
    def __init__(self):
        super().__init__()
        backbone = build_backbone(ResNet50_Weights)

    def forward(self, x):
        pass

def build_backbone(weight):
    backbone = resnet50(weight)

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