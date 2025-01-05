import torch
from torch import Tensor
from torchmetrics import KLDivergence, AUROC, CosineSimilarity, SpearmanCorrCoef
from torchmetrics.image import SpatialCorrelationCoefficient


@torch.no_grad()
def convert_kl_div(predict_img: Tensor, target_img: Tensor, epsilon: float = 1e-8) -> tuple[Tensor, Tensor]:
    predict_dist = torch.flatten(predict_img, start_dim=1) + 1
    target_dist = torch.flatten(target_img, start_dim=1) + 1

    predict_dist = predict_dist.softmax(dim=1) + epsilon
    target_dist = target_dist.softmax(dim=1) + epsilon

    return predict_dist, target_dist


@torch.no_grad()
def convert_auroc(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    predict_flat = (normalized(predict_img) > 0.5).float()
    target_flat = (normalized(target_img) > 0.5).float()
    return predict_flat, target_flat


@torch.no_grad()
def convert_sim(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    predict_img = predict_img.view(predict_img.size(0), -1)  # (B, C * W * H)
    target_img = target_img.view(target_img.size(0), -1)  # (B, C * W * H)

    predict_norm = torch.norm(predict_img, dim=1, keepdim=True)
    target_norm = torch.norm(target_img, dim=1, keepdim=True)
    return predict_img / predict_norm, target_img / target_norm


@torch.no_grad()
def convert_scc(predict_img: Tensor, target_img: Tensor) -> tuple[Tensor, Tensor]:
    predict_normalized = normalized(predict_img)
    target_normalized = normalized(target_img)

    return predict_normalized, target_normalized


@torch.no_grad()
def normalized(target: Tensor) -> Tensor:
    min_value = target.min()
    max_value = target.max()
    if min_value == max_value:
        return target

    # min と max の範囲に基づいて正規化
    result = (target - min_value) / (max_value - min_value)
    return result


def dummy_input(start=-1, end=1,shape: tuple = (32, 1, 256, 256)):
    b, c, h, w = shape
    return torch.linspace(start, end, steps=b * c * h * w).reshape(b, c, h, w)



if __name__ == '__main__':
    a1 = dummy_input()
    b1 = dummy_input()

    a1, b1 = convert_kl_div(a1, b1)
    print(f"KL Div: {KLDivergence()(torch.ones_like(a1), torch.ones_like(a1)):.2f}")

    a2 = dummy_input()
    b2 = dummy_input()

    a2, b2 = convert_auroc(a2, b2)
    print(f"AUROC: {AUROC(task="binary")(a2, b2).item():.2f}")

    a3 = dummy_input()
    b3 = dummy_input() + 1e-5

    a3, b3 = convert_sim(a3, b3)
    print(f"Sim: {CosineSimilarity(reduction="mean")(a3, b3).item():.2f}")

    a4 = dummy_input()
    b4 = dummy_input()

    a4, b4 = convert_scc(a4, b4)
    print(f"SCC: {SpatialCorrelationCoefficient()(a4, b4).item(): .2f}")
