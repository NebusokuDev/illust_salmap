from torchinfo import summary
from torch.nn import Module
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class DeepLabV3ForSaliency(Module):
    def __init__(self):
        super().__init__()
        # Pretrained DeepLabV3 with MobileNetV3 backbone
        self.deeplab = deeplabv3_mobilenet_v3_large(
            num_classes=1,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        # Freeze backbone layers
        for param in self.deeplab.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.deeplab(x)
        return output['out']


if __name__ == '__main__':
    model = DeepLabV3ForSaliency()
    summary(model, (3, 3, 256, 256))