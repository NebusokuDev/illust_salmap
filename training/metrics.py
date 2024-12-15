from abc import ABC, abstractmethod

import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch import Tensor, cosine_similarity
from torch.nn.functional import softmax, mse_loss


class Metrics(ABC):

    @abstractmethod
    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        pass

    def __call__(self, prediction: Tensor, correct: Tensor):
        return self.eval(prediction, correct)


class AreaUnderCurve(Metrics):
    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        # データを1次元配列に変換し、NumPy形式に
        prediction = prediction.view(-1).cpu().detach().numpy()
        correct = correct.view(-1).cpu().detach().numpy()

        # 正解ラベルをバイナリに変換
        correct_binary = (correct > 0.5).astype(int)

        # AUCスコアを計算
        return roc_auc_score(correct_binary, prediction)


class CorrelationCoefficient(Metrics):
    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        pred = prediction.view(-1)
        correct = correct.view(-1)
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(correct)
        numerator = torch.sum((pred - pred_mean) * (correct - target_mean))
        denominator = torch.sqrt(torch.sum((pred - pred_mean) ** 2) * torch.sum((correct - target_mean) ** 2))
        return (numerator / denominator).item()


class KLDivergence(Metrics):
    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction = softmax(prediction.view(-1), dim=0)
        correct = softmax(correct.view(-1), dim=0)
        return torch.sum(correct * torch.log(correct / (prediction + 1e-10))).item()


class NormalizedScanpathSaliency(Metrics):
    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction = prediction.view(-1)
        correct = correct.view(-1)

        # 実際の注目領域の平均と標準偏差を使って正規化
        correct = correct[correct > 0]
        pred_mean = torch.mean(prediction[correct > 0])
        pred_std = torch.std(prediction[correct > 0])

        # NSSスコア
        nss = (prediction - pred_mean) / (pred_std + 1e-10)
        return torch.mean(nss[correct > 0]).item()


class Similarity(Metrics):
    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction = prediction.view(-1)
        correct = correct.view(-1)
        return cosine_similarity(prediction, correct, dim=0).item()


class SmoothedAreaUnderCurve(Metrics):
    def __init__(self, sigma: float = 1):
        self.sigma = sigma

    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        pred_smooth = torch.tensor(gaussian_filter(prediction.cpu().numpy(), sigma=self.sigma))
        pred_smooth = pred_smooth.clone().detach().to(prediction.device)

        pred_smooth = pred_smooth.view(-1).cpu().detach().numpy()
        correct = correct.view(-1).cpu().detach().numpy()
        correct_binary = (correct > 0.5).astype(int)
        return roc_auc_score(correct_binary, pred_smooth)


class InformationGain(Metrics):
    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        prediction = softmax(prediction.view(-1), dim=0)
        correct = softmax(correct.view(-1), dim=0)
        return torch.sum(prediction * torch.log(correct / (prediction + 1e-10))).item()

class Dice(Metrics):

    def eval(self, prediction: Tensor, correct: Tensor) -> float:
        return 


if __name__ == '__main__':
    pred = torch.rand(16, 1, 256, 256)
    target = (torch.rand(16, 1, 256, 256))

    auc = AreaUnderCurve()
    cc = CorrelationCoefficient()
    kld = KLDivergence()
    nss = NormalizedScanpathSaliency()
    sim = Similarity()
    s_auc = SmoothedAreaUnderCurve()
    ig = InformationGain()

    print(f"MSE Score: {mse_loss(pred, target)}")
    print(f"AUC Score: {auc(pred, target)}")
    print(f"CC Score: {cc(pred, target)}")
    print(f"KLD Score: {kld(pred, target)}")
    print(f"NSS Score: {nss(pred, target)}")
    print(f"SIM Score: {sim(pred, target)}")
    print(f"sAUC Score: {s_auc(pred, target)}")
    print(f"IG Score: {ig(pred, target)}")
