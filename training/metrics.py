from abc import ABC, abstractmethod

import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch import Tensor, cosine_similarity
from torch.nn.functional import softmax


class Metrics(ABC):

    @abstractmethod
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        pass

    def __call__(self, prediction: Tensor, collect_label: Tensor):
        return self.eval(prediction, collect_label)


class AreaUnderCurve(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        pred = prediction.view(-1).cpu().detach().numpy()
        target = target.view(-1).cpu().detach().numpy()
        return roc_auc_score(target, pred)


class CorrelationCoefficient(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        pred = prediction.view(-1)
        target = target.view(-1)
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        numerator = torch.sum((pred - pred_mean) * (target - target_mean))
        denominator = torch.sqrt(torch.sum((pred - pred_mean) ** 2) * torch.sum((target - target_mean) ** 2))
        return (numerator / denominator).item()


class KLDivergence(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        prediction = softmax(prediction.view(-1), dim=0)
        target = softmax(target.view(-1), dim=0)
        return torch.sum(target * torch.log(target / (prediction + 1e-10))).item()


class NormalizedScanpathSaliency(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        prediction = prediction.view(-1)
        target = target.view(-1)

        # 実際の注目領域の平均と標準偏差を使って正規化
        target = target[target > 0]
        pred_mean = torch.mean(prediction[target > 0])
        pred_std = torch.std(prediction[target > 0])

        # NSSスコア
        nss = (prediction - pred_mean) / (pred_std + 1e-10)
        return torch.mean(nss[target > 0])


class Similarity(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        prediction = prediction.view(-1)
        target = target.view(-1)
        return cosine_similarity(prediction, target, dim=0).item()


class SmoothedAreaUnderCurve(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        pred_smooth = torch.tensor(gaussian_filter(prediction.cpu().numpy(), sigma=sigma))
        pred_smooth = torch.tensor(pred_smooth).to(prediction.device)
        pred_smooth = pred_smooth.view(-1).cpu().detach().numpy()
        target = target.view(-1).cpu().detach().numpy()
        return roc_auc_score(target, pred_smooth)


class InformationGain(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        # Placeholder for information gain implementation
        raise NotImplementedError
