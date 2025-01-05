import torch
from torchinfo import summary
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from illust_salmap.models.ez_bench import benchmark


def lraspp():
    model = lraspp_mobilenet_v3_large(num_classes=1,
                                      weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                                      aux_loss=True, )

    for param in model.backbone.parameters():
        param.requires_grad = False
    return model


if __name__ == '__main__':
    model = lraspp()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)

    print(model(torch.randn(shape)))
