from torch.nn import *
from torch.nn.functional import interpolate
from torchinfo import summary

from illust_salmap.models.ez_bench import benchmark


class UNetLite(Module):
    def __init__(self, in_channels=3, out_channels=1, use_skip_connection=True):
        super(UNetLite, self).__init__()
        self.use_skip_connection = use_skip_connection

        self.encoder_in_32 = Encoder(in_channels, 32)
        self.encoder_32_64 = Encoder(32, 64)
        self.encoder_64_128 = Encoder(64, 128)
        self.encoder_128_256 = Encoder(128, 256)
        self.encoder_256_512 = Encoder(256, 512)

        self.bottleneck = Bottleneck()

        self.decoder_512_512 = Decoder(512, 512, use_skip_connection)
        self.decoder_512_256 = Decoder(512, 256, use_skip_connection)
        self.decoder_256_128 = Decoder(256, 128, use_skip_connection)
        self.decoder_128_64 = Decoder(128, 64, use_skip_connection)
        self.decoder_64_32 = Decoder(64, 32, use_skip_connection)
        self.decoder_32_out = Decoder(32, out_channels, use_skip_connection)

        self.output = Tanh()

    def forward(self, x):
        # var name is "{layer}_{output_ch}"
        enc_32 = self.encoder_in_32(x)
        enc_64 = self.encoder_32_64(enc_32)
        enc_128 = self.encoder_64_128(enc_64)
        enc_256 = self.encoder_128_256(enc_128)
        enc_512 = self.encoder_256_512(enc_256)

        bottle_512 = self.bottleneck(enc_512)

        dec_512 = self.decoder_512_512.skip_connection(bottle_512, enc_512)
        dec_256 = self.decoder_512_256.skip_connection(dec_512, bottle_512)
        dec_128 = self.decoder_256_128.skip_connection(dec_256, enc_256)
        dec_64 = self.decoder_128_64.skip_connection(dec_128, enc_128)
        dec_32 = self.decoder_64_32.skip_connection(dec_64, enc_64)
        dec_out = self.decoder_32_out.skip_connection(dec_32, enc_32)

        return self.output(dec_out)


class Encoder(Module):
    def __init__(self, in_channels=3, out_channels=64, dropout_prob=0.1):
        super(Encoder, self).__init__()

        self.encoder = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            MaxPool2d(2, 2),
            BatchNorm2d(out_channels),
            LeakyReLU(0.2),
            Dropout2d(dropout_prob),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(Module):
    def __init__(self, in_channels=64, out_channels=3, use_skip_connection=True, dropout_prob=0):
        super(Decoder, self).__init__()

        self.use_skip_connection = use_skip_connection

        self.decoder = Sequential(
            ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2),
            Conv2d(in_channels, out_channels, 3, 1, 1),
            BatchNorm2d(out_channels),
            Dropout2d(dropout_prob),
        )

    def forward(self, x):
        return self.decoder(x)

    def skip_connection(self, x, y):
        if not self.use_skip_connection:
            return self.forward(x)

        y = interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)

        return self.forward(x + y)


class Bottleneck(Module):
    def __init__(self, channels=512, dropout_prob=0.3):
        super(Bottleneck, self).__init__()

        self.bottleneck = Sequential(
            Conv2d(channels, channels, 4, 2, 1),
            BatchNorm2d(channels),
            LeakyReLU(),
            Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.bottleneck(x)


if __name__ == '__main__':
    model = UNetLite()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
