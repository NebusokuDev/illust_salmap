from abc import ABC, abstractmethod

import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch import Tensor, cosine_similarity
from torch.nn.functional import softmax


class Metrics(ABC):

    @abstractmethod
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        pass

    def __call__(self, prediction: Tensor, correct: Tensor):
        with torch.no_grad():
            return self._eval(prediction, correct)


class PixelWiseAccuracy(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction_binary = (prediction > 0.5).float()
        correct_binary = (correct > 0.5).float()
        return (prediction_binary == correct_binary).float().mean().item()


class JaccardIndex(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction = (prediction > 0.5)
        correct = (correct > 0.5)

        intersection = (prediction & correct).float().sum((1, 2))
        union = (prediction | correct).float().sum((1, 2))

        iou = intersection / (union + 1e-6)
        return iou.mean().item()


class AreaUnderCurve(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        # データを1次元配列に変換し、NumPy形式に
        prediction = prediction.view(-1).numpy()
        correct = correct.view(-1).numpy()

        # 正解ラベルをバイナリに変換
        correct_binary = (correct > 0.5).astype(int)

        # AUCスコアを計算
        return roc_auc_score(correct_binary, prediction)


class CorrelationCoefficient(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        pred = prediction.view(-1)
        correct = correct.view(-1)
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(correct)
        numerator = torch.sum((pred - pred_mean) * (correct - target_mean))
        denominator = torch.sqrt(torch.sum((pred - pred_mean) ** 2) * torch.sum((correct - target_mean) ** 2))
        return (numerator / denominator).item()


class KLDivergence(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction = softmax(prediction.view(-1), dim=0)
        correct = softmax(correct.view(-1), dim=0)
        return torch.sum(correct * torch.log(correct / (prediction + 1e-10))).item()


class NormalizedScanpathSaliency(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        # サイズが一致しているか確認
        if prediction.shape != correct.shape:
            raise ValueError(f"Prediction and correct have different shapes: {prediction.shape} vs {correct.shape}")

        prediction = prediction.view(-1)
        correct = correct.view(-1)

        # 実際の注目領域の平均と標準偏差を使って正規化
        correct_mask = correct > 0  # 注目領域をマスク
        if correct_mask.sum() == 0:  # 注目領域がない場合
            return 0.0  # NSSスコアは計算できないので0を返す

        # サイズが一致することを確認した後、インデックス操作を行う
        pred_mean = torch.mean(prediction[correct_mask])
        pred_std = torch.std(prediction[correct_mask])

        # NSSスコア
        nss = (prediction - pred_mean) / (pred_std + 1e-10)
        return torch.mean(nss[correct_mask]).item()


class Similarity(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction = prediction.view(-1)
        correct = correct.view(-1)
        return cosine_similarity(prediction, correct, dim=0).item()


class SmoothedAreaUnderCurve(Metrics):
    def __init__(self, sigma: float = 1):
        self.sigma = sigma

    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        pred_smooth = torch.tensor(gaussian_filter(prediction.numpy(), sigma=self.sigma))
        pred_smooth = pred_smooth.clone().detach().to(prediction.device)

        pred_smooth = pred_smooth.view(-1).numpy()
        correct = correct.view(-1).numpy()
        correct_binary = (correct > 0.5).astype(int)
        return roc_auc_score(correct_binary, pred_smooth)


class InformationGain(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction = softmax(prediction.view(-1), dim=0)
        correct = softmax(correct.view(-1), dim=0)
        return torch.sum(prediction * torch.log(correct / (prediction + 1e-10))).item()


class Dice(Metrics):
    def _eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction_binary = (prediction > 0.5).float()
        correct_binary = (correct > 0.5).float()
        intersection = (prediction_binary * correct_binary).sum(dim=[1, 2, 3])
        dice = (2 * intersection) / (prediction_binary.sum(dim=[1, 2, 3]) + correct_binary.sum(dim=[1, 2, 3]) + 1e-6)
        return dice.mean().item()


if __name__ == '__main__':
    pred = torch.randn(16, 1, 256, 256)
    target = (torch.randn(16, 1, 256, 256))

    pixel = PixelWiseAccuracy()
    jaccard = JaccardIndex()
    auc = AreaUnderCurve()
    cc = CorrelationCoefficient()
    kld = KLDivergence()
    nss = NormalizedScanpathSaliency()
    sim = Similarity()
    s_auc = SmoothedAreaUnderCurve()
    ig = InformationGain()
    dice = Dice()

    print(f"pixel Score: {pixel(pred, target)}")
    print(f"IoU Score: {jaccard(pred, target)}")
    print(f"AUC Score: {auc(pred, target)}")
    print(f"CC Score: {cc(pred, target)}")
    print(f"KLD Score: {kld(pred, target)}")
    print(f"NSS Score: {nss(pred, target)}")
    print(f"SIM Score: {sim(pred, target)}")
    print(f"sAUC Score: {s_auc(pred, target)}")
    print(f"IG Score: {ig(pred, target)}")
    print(f"DICE Score: {dice(pred, target)}")
