from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.functional import auroc


# それぞれ実装してほしいです
# torchmetrics に実装されている場合は削除してください


class AUCJudd(Metric):
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.labels = []

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.predictions.append(preds)
        self.labels.append(target)

    def compute(self) -> torch.Tensor:
        predictions = torch.cat(self.predictions, dim=0)
        labels = torch.cat(self.labels, dim=0)

        # AUC計算
        auc_score = auroc(labels, predictions, task="binary")
        return torch.tensor(auc_score)


class AUCBorji(Metric):
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.ground_truth = []

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.predictions.append(preds)
        self.ground_truth.append(target)

    def compute(self) -> torch.Tensor:
        predictions = torch.cat(self.predictions, dim=0)
        ground_truth = torch.cat(self.ground_truth, dim=0)

        auc_score = auroc(ground_truth, predictions, task="binary")
        return torch.tensor(auc_score)


class NormalizedScanpathSaliency(Metric):
    def __init__(self):
        super().__init__()
        self.scanpaths = []
        self.saliency_maps = []

    def update(self, scanpath: torch.Tensor, saliency_map: torch.Tensor) -> None:
        self.scanpaths.append(scanpath)
        self.saliency_maps.append(saliency_map)

    def compute(self) -> torch.Tensor:
        scanpaths = torch.cat(self.scanpaths, dim=0)
        saliency_maps = torch.cat(self.saliency_maps, dim=0)

        # サリエンシーマップと視線の経路の類似度を計算する例（ノーマライズ）
        similarity = torch.cosine_similarity(scanpaths, saliency_maps, dim=-1)
        return similarity.mean()


class InformationGain(Metric):
    def update(self, *_: Any, **__: Any) -> None:
        pass

    def compute(self) -> Any:
        pass
