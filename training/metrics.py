import torch
from torch import Tensor
from torch.nn.functional import normalize
from torchmetrics import KLDivergence, AUROC, CosineSimilarity
from torchmetrics.image import SpatialCorrelationCoefficient


# torchmetricsに入力するための変換を行う関数を実装
def convert_kl_div(predict_img: Tensor, target_img: Tensor, epsilon=1e-8) -> tuple[Tensor, Tensor]:
    def normalize_to_distribution(tensor: Tensor) -> Tensor:
        # バッチ次元を考慮して、全体を1に正規化
        tensor = normalized(tensor) + epsilon  # 先にepsilonを加えることでゼロを防ぐ
        return tensor / tensor.sum(dim=-1, keepdim=True)  # バッチ次元を除外して正規化

    # 確率分布に変換
    predict_dist = normalize_to_distribution(predict_img)
    target_dist = normalize_to_distribution(target_img)

    # 対数分布を計算し、バッチ次元を追加
    predict_log_dist = predict_dist.log().flatten(start_dim=1)
    target_dist = target_dist.flatten(start_dim=1)

    return predict_log_dist, target_dist



def convert_auroc(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    predict_flat = (normalized(predict_img) > 0.5).float()
    target_flat = (normalized(target_img) > 0.5).float()
    return predict_flat, target_flat


def convert_sim(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    # 各テンソルを(3, 256*256)にフラット化
    predict_img = predict_img.view(predict_img.size(0), -1)  # (3, 65536)
    target_img = target_img.view(target_img.size(0), -1)  # (3, 65536)
    # 正規化: L2ノルムによる正規化
    predict_img = normalize(predict_img, p=2, dim=1)
    target_img = normalize(target_img, p=2, dim=1)
    return predict_img, target_img


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


if __name__ == '__main__':
    a1 = torch.randn(3, 1, 256, 256)
    b1 = torch.randn(3, 1, 256, 256)

    a1, b1 = convert_kl_div(a1, b1)
    print(KLDivergence()(a1, b1))

    a2 = torch.randn(3, 1, 256, 256)
    b2 = torch.randn(3, 1, 256, 256)

    a2, b2 = convert_auroc(a2, b2)
    print(AUROC(task="binary")(a2, b2))

    a3 = torch.randn(3, 1, 256, 256)
    b3 = torch.randn(3, 1, 256, 256)

    a3, b3 = convert_sim(a3, b3)
    print(CosineSimilarity()(a3, b3))

    a4 = torch.randn(3, 1, 256, 256)
    b4 = torch.randn(3, 1, 256, 256)

    a4, b4 = convert_scc(a4, b4)
    print(SpatialCorrelationCoefficient()(a4, b4))

