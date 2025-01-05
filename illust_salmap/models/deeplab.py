from torch.nn import MSELoss, Module
from torchinfo import summary
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from illust_salmap.models.ez_bench import benchmark


def deeplab():
    deeplab = deeplabv3_mobilenet_v3_large(num_classes=1,
                                           weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                                           aux_loss=True, )

    for param in deeplab.backbone.parameters():
        param.requires_grad = False
    return deeplab


class DeepLabLoss(Module):
    def __init__(self, aux_weight=0.3, criterion=MSELoss()):
        super().__init__()
        self.criterion = criterion
        self.aux_weight = aux_weight

    def forward(self, pred, target):
        main = self.criterion(pred['out'], target)
        aux = self.criterion(pred['aux'], target)

        return main + self.aux_weight * aux


if __name__ == '__main__':
    model = deeplab()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
