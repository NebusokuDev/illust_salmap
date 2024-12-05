from torch.nn import Module
from torch.nn.functional import pad
from torch import Tensor


class ImageShapeAdjuster(Module):
    def __init__(self):
        super(ImageShapeAdjuster, self).__init__()
        self.padding_size_lrtb = (0, 0, 0, 0)  # デフォルトのパディングサイズ

    def forward(self, target: Tensor):
        return self.pad(target)

    def pad(self, target: Tensor):
        self.padding_size_lrtb = self.get_padding_size(target)
        return pad(target, self.padding_size_lrtb)  # パディング関数を変更

    def crop(self, target: Tensor):
        # 保存された `padding_size` を使ってクロップ
        left, right, top, bottom = self.padding_size_lrtb
        height, width = target.shape[2], target.shape[3]

        return target[:, :, top:height - bottom, left:width - right]

    def get_padding_size(self, target: Tensor):
        height, width = target.shape[2], target.shape[3]

        if height > width:
            left = (height - width) // 2
            right = height - width - left
            top, bottom = 0, 0
        else:
            top = (width - height) // 2
            bottom = width - height - top
            left, right = 0, 0

        return left, right, top, bottom