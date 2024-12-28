import torch
from torchinfo import summary
from torch.nn import Module, MSELoss
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, deeplabv3_resnet101


def deeplab():
    deeplab = deeplabv3_mobilenet_v3_large(
        num_classes=1,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )

    for param in deeplab.backbone.parameters():
        param.requires_grad = False
    return deeplab


class DeepLabLoss(Module):
    def __init__(self, criterion=MSELoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, pred, target):
        main = pred['out']
        aux = pred['aux']

        return self.criterion(pred, target)


if __name__ == '__main__':
    model = deeplab()

    criterion = DeepLabLoss()
    summary(model, (3, 3, 256, 256))
