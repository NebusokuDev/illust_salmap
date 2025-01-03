import torch
from torch.ao.nn.quantized import ConvTranspose2d
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d, Tanh
from torchinfo import summary
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class SalGANGenerator(Module):
    def __init__(self, head=Tanh()):
        super().__init__()
        backbone = vgg16_bn(VGG16_BN_Weights.IMAGENET1K_V1)

        encoder_fist: Module = backbone.features[:17]
        for param in encoder_fist.parameters():
            param.requires_grad = False

        encoder_last = backbone.features[17:-1]

        self.encoder = Sequential(
            encoder_fist,
            encoder_last,
        )

        self.decoder = Sequential(
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 1)
        )

        self.head = head

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


class DecoderBlock(Module):
    def __init__(self, in_channels, out_channels, activation=LeakyReLU()):
        super().__init__()
        self.upsample = ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.conv1 = Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.batch_norm1 = BatchNorm2d(in_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    model = SalGANGenerator()
    output = model(torch.ones(1, 3, 256, 256))
    summary(model, (3, 256, 256))
