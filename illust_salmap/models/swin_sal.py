from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, Tanh
from torchinfo import summary
from torchvision.models import swin_v2_t, Swin_V2_T_Weights


class SwinSal(Module):
    def __init__(self, head=Tanh()):
        super().__init__()
        backbone = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)

        # Encoder: SwinV2 Backbone
        self.encoder = Sequential(
            backbone.features,
            backbone.norm,
            backbone.permute,
        )

        # Decoder: Series of DecoderBlocks
        self.decoder = Sequential(
            DecoderBlock(768, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 1),
        )

        self.head = head

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x


class DecoderBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_transposed = ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.conv = Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_transposed(x)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    model = SwinSal()
    summary(model, (4, 3, 512, 512))
