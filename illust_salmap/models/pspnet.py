from torch.nn import Module, Conv2d, BatchNorm2d
from torchinfo import summary


class PSPNet(Module):
    def __init__(self):
        super().__init__()


class FeatureMap(Module):
    pass


class PyramidPool(Module):
    pass


class AUXLoss(Module):
    pass


class Upsample(Module):
    pass


class Decoder(Module):
    pass

class ResNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.conv2 = Conv2d(64, 64, 3, 1, 1, bias=False)

if __name__ == '__main__':
    summary(PSPNet(), (4, 3, 128, 128))
