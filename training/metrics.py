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


class RocAucMetric(Module):
    def forward(self, saliency_map, ground_truth):
        # 予測とラベルを1次元に変換
        saliency_map_flat = saliency_map.view(-1)
        ground_truth_flat = ground_truth.view(-1)

        # 並べ替えのためのインデックス取得
        sorted_indices = torch.argsort(saliency_map_flat, descending=True)
        sorted_truth = ground_truth_flat[sorted_indices]

        # 真陽性累積和と偽陽性累積和の計算
        tpr = torch.cumsum(sorted_truth, dim=0) / sorted_truth.sum()  # 真陽性率
        fpr = torch.cumsum(1 - sorted_truth, dim=0) / (1 - sorted_truth).sum()  # 偽陽性率

        # ROC曲線の面積 (AUC)
        auc = torch.trapz(tpr, fpr)
        return auc



class Precision(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        predictions = (predictions > 0.5).float()  # 閾値を適用してバイナリ化
        true_positive = (predictions * targets).sum()
        predicted_positive = predictions.sum()
        precision = true_positive / (predicted_positive + 1e-8)  # ゼロ割防止
        return precision


class Recall(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        predictions = (predictions > 0.5).float()  # 閾値を適用してバイナリ化
        true_positive = (predictions * targets).sum()
        actual_positive = targets.sum()
        recall = true_positive / (actual_positive + 1e-8)  # ゼロ割防止
        return recall


class F1Score(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.precision = Precision()
        self.recall = Recall()

    def forward(self, predictions, targets):
        precision = self.precision(predictions, targets)
        recall = self.recall(predictions, targets)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # ゼロ割防止
        return f1
