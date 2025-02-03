from pathlib import Path

import torch
from torch.nn import Module
from torchinfo import summary
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.mobilenet import MobileNet_V3_Large_Weights

from illust_salmap.models.ez_bench import benchmark
from illust_salmap.training.saliency_model import SaliencyModel


class DeepLab(Module):
    def __init__(self):
        super().__init__()
        deeplab = deeplabv3_mobilenet_v3_large(weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                                               num_classes=1,
                                               aux_loss=True, )
        self.model = deeplab

    def forward(self, x):
        if self.training:
            result = self.model(x)
            pred = result['out']
            aux = result['aux']
            return pred, aux

        return self.model(x)['out']


def deeplab(ckpt_path: str | Path = None):
    model = SaliencyModel(DeepLab())
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)['state_dict']
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    ckpt_path = input("ckpt path: ").strip("'").strip('"')
    model = deeplab(ckpt_path)
    model.train()

    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
