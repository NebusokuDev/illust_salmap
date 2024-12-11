import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch import Tensor
from torch.nn import Module


def normalize01(tensor):
    """-1~1の範囲を0~1に正規化"""
    return (tensor + 1) / 2


def binarize_map(saliency_map, threshold=0.5):
    return (saliency_map > threshold).int()


class JaccardIndex(Module):
    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        saliency_map = normalize01(saliency_map)
        ground_truth = normalize01(ground_truth)
        intersection = torch.logical_and(saliency_map >= self.threshold, ground_truth >= self.threshold).sum()
        union = torch.logical_or(saliency_map >= self.threshold, ground_truth >= self.threshold).sum()
        return intersection / union if union != 0 else torch.tensor(0.0)


class PixelWiseAccuracy(Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        saliency_map = normalize01(saliency_map)
        ground_truth = normalize01(ground_truth)
        pixel_errors = torch.abs(saliency_map - ground_truth)

        correct_pixels = pixel_errors <= self.threshold

        accuracy = correct_pixels.float().mean()

        return accuracy


class MAE(Module):
    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        saliency_map = normalize01(saliency_map)
        ground_truth = normalize01(ground_truth)
        return torch.mean(torch.abs(saliency_map - ground_truth))


class MSE(Module):
    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        saliency_map = normalize01(saliency_map)
        ground_truth = normalize01(ground_truth)
        return torch.mean((saliency_map - ground_truth) ** 2)


class AUC(Module):
    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        saliency_map = binarize_map(saliency_map.detach())
        ground_truth = binarize_map(ground_truth.detach())

        saliency_map_flat = saliency_map.view(-1).cpu().numpy()
        ground_truth_flat = ground_truth.view(-1).cpu().numpy()

        return roc_auc_score(ground_truth_flat, saliency_map_flat)


class Precision(Module):
    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        saliency_map = binarize_map(saliency_map.detach())
        ground_truth = binarize_map(ground_truth.detach())

        saliency_map = normalize01(saliency_map).view(-1).cpu().numpy()
        ground_truth = normalize01(ground_truth).view(-1).cpu().numpy()
        # Precision は適合率（正解したピクセルの割合）
        return precision_score(ground_truth, saliency_map > 0.5)


class Recall(Module):
    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        saliency_map = binarize_map(saliency_map.detach())
        ground_truth = binarize_map(ground_truth.detach())

        saliency_map = normalize01(saliency_map).view(-1).cpu().numpy()
        ground_truth = normalize01(ground_truth).view(-1).cpu().numpy()
        # Recall は再現率（サリエンシーの高い領域が正解領域にどれだけ一致しているか）
        return recall_score(ground_truth, saliency_map > 0.5)


class F1Score(Module):
    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        saliency_map = binarize_map(saliency_map.detach())
        ground_truth = binarize_map(ground_truth.detach())

        saliency_map = normalize01(saliency_map).view(-1).cpu().numpy()
        ground_truth = normalize01(ground_truth).view(-1).cpu().numpy()

        # F1スコアはPrecisionとRecallの調和平均
        return f1_score(ground_truth, saliency_map > 0.5)
