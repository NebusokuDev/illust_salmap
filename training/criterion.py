import torch
from torch import nn


class SaliencyLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def compute_centroid(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # S: (B, C, H, W) -> バッチサイズ、チャンネル、高さ、幅
        B, C, H, W = S.shape
        device = S.device

        # 座標の計算
        x_coords = torch.arange(W, device=device).view(1, 1, 1, -1).expand(B, C, H, W)
        y_coords = torch.arange(H, device=device).view(1, 1, -1, 1).expand(B, C, H, W)

        # 各チャンネルにおける重心を計算
        total = S.sum(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
        cx = (x_coords * S).sum(dim=(-2, -1), keepdim=True) / (total + 1e-6)
        cy = (y_coords * S).sum(dim=(-2, -1), keepdim=True) / (total + 1e-6)
        return cx, cy

    def forward(self, predict: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        # バッチ内の各サンプルに対して損失を計算
        B, C, H, W = predict.shape

        # 各バッチに対して損失を計算
        centroid_distances = []
        concentration_losses = []

        for i in range(B):
            # 各画像ごとの予測とグラウンドトゥルースの重心のズレ
            cx1, cy1 = self.compute_centroid(predict[i:i + 1])
            cx2, cy2 = self.compute_centroid(ground_truth[i:i + 1])
            centroid_distance = torch.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2).mean()

            # 濃度の損失
            x_coords = torch.arange(W, device=predict.device).view(1, -1).expand(H, W)
            y_coords = torch.arange(H, device=predict.device).view(-1, 1).expand(H, W)
            cx, cy = self.compute_centroid(predict[i:i + 1])
            distances = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
            concentration_loss = (distances * predict[i:i + 1]).sum() / (predict[i:i + 1].sum() + 1e-6)

            centroid_distances.append(centroid_distance)
            concentration_losses.append(concentration_loss)

        # バッチ内の平均損失を計算
        centroid_loss_batch = torch.stack(centroid_distances).mean()
        concentration_loss_batch = torch.stack(concentration_losses).mean()

        # 総損失
        return self.alpha * centroid_loss_batch + self.beta * concentration_loss_batch
