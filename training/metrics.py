from abc import ABC, abstractmethod

import torch
from torch import Tensor, cosine_similarity
from torch.nn.functional import kl_div
from torcheval.metrics.functional import auc


class Metrics(ABC):

    @abstractmethod
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        pass

    def __call__(self, prediction: Tensor, collect_label: Tensor):
        return self.eval(prediction, collect_label)


class AreaUnderCurve(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        prediction = prediction.cpu().detach().numpy()
        return auc(prediction, target).item()


class CorrelationCoefficient(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        pred = prediction.view(-1)
        target = target.view(-1)
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        numerator = torch.sum((pred - pred_mean) * (target - target_mean))
        denominator = torch.sqrt(torch.sum((pred - pred_mean) ** 2) * torch.sum((target - target_mean) ** 2))
        return numerator / denominator


class KLDivergence(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        return kl_div(prediction.log(), target, reduction="batchmean").item()


class NormalizedScanpathSaliency(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        # Placeholder, depends on your task and specific formula for this metric
        raise NotImplementedErro

class Similarity(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        return cosine_similarity(prediction.view(1, -1), target.view(1, -1)).item()



class SmoothedAreaUnderCurve(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        smoothed_prediction = torch.conv1d(prediction.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, 5) / 5, padding=2)  # Simple smoothing
        smoothed_prediction = smoothed_prediction.squeeze()
        return auc(smoothed_prediction.cpu().detach().numpy(), target).item()


class InformationGain(Metrics):
    def eval(self, prediction: Tensor, target: Tensor) -> float:
        # Placeholder for information gain implementation
        raise NotImplementedError
