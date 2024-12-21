import torch
from torch import Tensor
from torchmetrics import Metric


# torchmetricsに入力するための変換を行う関数を実装
def convert_kl_div(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    def normalize_to_distribution(tensor: Tensor) -> Tensor:
        tensor = tensor / tensor.sum()  # 全体を1に正規化
        return tensor

    # 確率分布に変換
    predict_dist = normalize_to_distribution(predict_img)
    target_dist = normalize_to_distribution(target_img)

    # 対数分布を計算し、バッチ次元を追加
    predict_log_dist = predict_dist.unsqueeze(0).log()
    target_dist = target_dist.unsqueeze(0)

    return predict_log_dist, target_dist


def convert_auroc(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    predict_flat = predict_img.view(-1)
    target_flat = (target_img < 0.5).view(-1)

    return predict_flat, target_flat


def convert_sim(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    # 類似度の計算用に正規化してから計算
    predict_normalized = normalized(predict_img)
    target_normalized = normalized(target_img)

    return predict_normalized, target_normalized


def convert_scc(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    # 正規化してから相関係数の計算
    predict_normalized = normalized(predict_img)
    target_normalized = normalized(target_img)

    return predict_normalized, target_normalized


def normalized(target: Tensor) -> Tensor:
    min_value = target.min()
    max_value = target.max()

    # min と max の範囲に基づいて正規化
    result = (target - min_value) / (max_value - min_value)
    return result


class NormalizedScanpathSaliency(Metric):
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        return torch.zeros()
