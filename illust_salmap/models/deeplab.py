from torch.nn import Module
from torchinfo import summary
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from illust_salmap.models.ez_bench import benchmark


class DeepLab(Module):
    def __init__(self):
        super().__init__()
        deeplab = deeplabv3_mobilenet_v3_large(num_classes=1, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2, aux_loss=True,)
        self.model = deeplab

    def forward(self, x):
        if self.training:
            result = self.model(x)
            pred = result['out']
            aux = result['aux']
            return pred, aux

        return self.model(x)['out']

if __name__ == '__main__':
    model = DeepLab()
    model.train()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
