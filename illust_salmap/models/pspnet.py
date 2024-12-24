from torch.nn import Module
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


if __name__ == '__main__':
    summary(PSPNet(), (4, 3, 128, 128))
