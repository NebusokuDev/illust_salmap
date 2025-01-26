import torch
from torch.nn import Module
from torchinfo import summary
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from illust_salmap.models.ez_bench import benchmark
from illust_salmap.training.saliency_model import SaliencyModel


class LRASPP(Module):
    def __init__(self, freeze_backbone=True, ):
        super().__init__()
        model = lraspp_mobilenet_v3_large(num_classes=1,
                                          weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2, )
        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
        self.model = model

    def forward(self, x):
        return self.model(x)['out']


def lrasppp(ckpt_path=None):
    model = SaliencyModel(LRASPP())
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)['state_dict']
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    model = LRASPP()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
