from torch.nn import Module
from torchinfo import summary
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from illust_salmap.models.ez_bench import benchmark


class LRASPP(Module):
    def __init__(self, freeze_backbone=True,):
        super().__init__()
        model = lraspp_mobilenet_v3_large(num_classes=1,
                                          weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,)
        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
        self.model = model

    def forward(self, x):
        return self.model(x)['out']


if __name__ == '__main__':
    model = LRASPP()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
