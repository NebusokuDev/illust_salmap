import torch
from torch.nn import *
from torchvision.ops import Permute
from torchvision.models.swin_transformer import SwinTransformerBlockV2, PatchMergingV2
from torchsummary import summary


class SwinUNet(Module):
    def __init__(self, classes=10, in_channels=3):
        super().__init__()

        self.patch_partition = Sequential(
            Conv2d(in_channels, 64, kernel_size=1, stride=1),
            Permute([0, 2, 3, 1]),
            LayerNorm(64)
        )

        self.encoder1 = EncoderBlock(64, window_size=4, shift_size=4)
        self.encoder2 = EncoderBlock(64, window_size=4, shift_size=4)
        self.encoder3 = EncoderBlock(128)
        self.encoder4 = EncoderBlock(128)

    def forward(self, x):
        # エンコード部分
        patches = self.patch_partition(x)  # ダウンサンプルされた画像
        e = self.encoder1(patches)
        e = self.encoder2(e)
        # e = self.encoder3(e)
        # e = self.encoder4(e)
        return e


class EncoderBlock(Module):
    def __init__(self, in_channels, window_size=4, shift_size=8):
        super().__init__()
        self.block = Sequential(
            SwinTransformerBlockV2(dim=in_channels, num_heads=4, window_size=[window_size, window_size], shift_size=[shift_size, shift_size]),
            PatchMergingV2(dim=in_channels)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(Module):
    def __init__(self):
        super().__init__()
        self.block = Sequential(
            SwinTransformerBlockV2(dim=64, num_heads=4, window_size=[4, 4], shift_size=[8, 8])
        )

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    model = SwinUNet(classes=1)

    # print(model)
    output = model(torch.randn(1, 3, 512, 512))
    print(output.shape)
    # summary(model, (3, 256, 256))  # 入力サイズ(3, 256, 256)でモデルの要約を表示
