from torch import Tensor
from torch.nn import ConvTranspose2d, Sequential, Module, ReLU
from torch.nn.functional import interpolate
from torchsummary import summary
from torchvision.models import swin_v2_t, Swin_V2_T_Weights


class SwinSeg(Module):
    def __init__(self, num_classes):
        super(SwinSeg, self).__init__()
        # Backbone
        backbone = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        self.features = backbone.features

        self.decoder = Sequential(
            ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: Tensor, target_size=None):
        size = target_size or x.shape[-2:]
        x = self.features(x)
        x = x.permute(0, 3, 1, 2)
        x = self.decoder(x)
        return interpolate(x, size=size, mode="bilinear", align_corners=False)


if __name__ == '__main__':
    model = SwinSeg(3)
    print(model)
    summary(model, (3, 256, 256))
