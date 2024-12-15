import torch
from torch import Tensor
from torch.nn import Module, MSELoss


class DiceLoss(Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        # 予測とターゲットを1次元化
        predict = torch.flatten(predict)
        target = torch.flatten(target)

        # 交差部分（intersection）の計算
        intersection = torch.sum(predict * target)

        # Dice Lossの計算
        return 1 - (2. * intersection + self.smooth) / (torch.sum(predict) + torch.sum(target) + self.smooth)


class MSEDiceLoss(Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.mse = MSELoss()
        self.dice = DiceLoss()
        self.alpha = alpha  # MSELoss の重み
        self.beta = beta  # DiceLoss の重み

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        mse_loss = self.mse(predict, target)
        dice_loss = self.dice(predict, target)

        # MSE Loss と Dice Loss を alpha と beta でスケーリング
        return (self.alpha * mse_loss) * (self.beta * dice_loss)