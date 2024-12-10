import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.nn import Module


class NSS(Module):
    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        return torch.mean(torch.abs(saliency_map - ground_truth))


class AUC(Module):
    def forward(self, saliency_map: Tensor, ground_truth: Tensor):
        # 予測をフラットにし、正解もフラットにする
        saliency_map_flat = saliency_map.view(-1).detach().cpu().numpy()
        ground_truth_flat = ground_truth.view(-1).detach().cpu().numpy()

        # AUCを計算
        return roc_auc_score(ground_truth_flat, saliency_map_flat)