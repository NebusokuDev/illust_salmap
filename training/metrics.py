import torch
from torch import Tensor, softmax
from torchmetrics import Metric, KLDivergence, CosineSimilarity, AUROC
from torchmetrics.image import SpatialCorrelationCoefficient
from torchmetrics.wrappers import LambdaInputTransformer, BinaryTargetTransformer


# 画像のkl_divを評価したい
def build_kl_div() -> Metric:
    return LambdaInputTransformer(
        wrapped_metric=KLDivergence(),
        transform_pred=lambda x: softmax(x.flatten(start_dim=1), dim=1),
        transform_target=lambda x: softmax(x.flatten(start_dim=1), dim=1),
    )


def build_auroc() -> Metric:
    return BinaryTargetTransformer(
        wrapped_metric=AUROC("binary")
    )


def build_sim() -> Metric:
    return LambdaInputTransformer(
        CosineSimilarity(),
        transform_pred=lambda x: x.flatten(start_dim=1),
        transform_target=lambda x: x.flatten(start_dim=1),
    )


def build_scc() -> Metric:
    return LambdaInputTransformer(
        wrapped_metric=SpatialCorrelationCoefficient(),
        transform_pred=normalized,
        transform_target=normalized
    )


def normalized(target: Tensor) -> Tensor:
    min_value = target.min()
    max_value = target.max()

    # min と max の範囲に基づいて正規化
    result = (target - min_value) / (max_value - min_value)
    return result


class AUCJudd(Metric):
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        raise NotImplementedError


class AUCBorji(Metric):
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        return torch.zeros()


class NormalizedScanpathSaliency(Metric):
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        return torch.zeros()

if __name__ == '__main__':
    # テストデータ
    metric = build_sim()
    for i in range(10):
        predict = torch.randn(32, 1, 384, 256)
        target = torch.ones(32, 1, 384, 256)

        # 正規化とリシェイプの確認

        metric(predict, target)
    print(metric.compute())
