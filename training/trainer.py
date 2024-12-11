from csv import DictWriter
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from matplotlib import pyplot


class Trainer:
    def __init__(self,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 criterion: Module,
                 device: torch.device,
                 model_name: str,
                 batch_stride: int = 3,
                 metrics: Optional[Dict[str, Callable[[Tensor, Tensor], float]]] = None,
                 metrics_score: Callable[[list[Dict[str, float]]], float] = None,
                 log_root: str = "./logs",
                 model_root: str = "./trained_model",
                 date_format: str = "%Y_%m_%d/%H_%M"
                 ):
        timestamp = get_timestamp(date_format)

        self.device = device
        self.batch_stride = batch_stride
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.metrics = metrics or {}
        self.log_root = Path(log_root).resolve() / model_name / timestamp
        self.model_root = Path(model_root).resolve() / model_name / timestamp
        self.summary_writer = SummaryWriter(str(self.log_root / "tensorboard"))

        self.metrics_score = metrics_score or default_metric_score

        self.model_name = model_name

    def _train(self, epoch, model: Module, optimizer: Optimizer):
        model.train()
        report = []
        for batch_idx, (image, label) in enumerate(self.train_dataloader):
            image, label = image.to(self.device), label.to(self.device)
            optimizer.zero_grad()
            predict = model(image)
            loss = self.criterion(predict, label)
            loss.backward()
            optimizer.step()

            if batch_idx % self.batch_stride == 0:
                metrics = self._eval_metrics(epoch, batch_idx, predict, label, loss)
                report.append(metrics)

        return report

    def _test(self, epoch, model: Module):
        model.eval()
        report = []
        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(self.test_dataloader):
                image, label = image.to(self.device), label.to(self.device)

                predict = model(image)
                loss = self.criterion(predict, label)

                if batch_idx % self.batch_stride == 0:
                    metrics = self._eval_metrics(epoch, batch_idx, predict, label, loss)
                    report.append(metrics)

        return report

    def _eval_metrics(self, epoch, batch, predict: Tensor, label: Tensor, loss: Tensor):
        metrics = {
            "epoch": epoch,
            "batch": batch,
            "loss": loss.item()
        }

        self.summary_writer.add_scalar("loss", loss.item())

        with torch.no_grad():
            for metric_label, metric_fn in self.metrics.items():
                metric = metric_fn(predict.detach(), label.detach())
                self.summary_writer.add_scalar(metric_label, metric.item())
                metrics[metric_label] = metric.item()
            formatted_metrics = "\t".join(f"{key}: {value:>8.4g}" for key, value in metrics.items())
            print(formatted_metrics)

        return metrics

    def _visualize(self, model: torch.nn.Module, epoch):
        # テストデータローダーからバッチを取得
        image, label = next(iter(self.test_dataloader))
        image, label = image.to(self.device), label.to(self.device)

        # モデルによる予測
        with torch.no_grad():
            predict = model(image)

        # TensorBoardに画像をログ
        self.summary_writer.add_image("visualize/image", make_grid(image), global_step=epoch)
        self.summary_writer.add_image("visualize/prediction", make_grid(predict), global_step=epoch)

        # Matplotlibで可視化
        fig, axes = pyplot.subplots(1, 3, figsize=(12, 4))

        # 入力画像
        axes[0].imshow(image[0].cpu().permute(1, 2, 0).numpy())
        axes[0].set_title("Input Image")
        axes[0].set_axis_off()

        # 正解ラベル
        axes[1].imshow(label[0].cpu().permute(1, 2, 0).numpy())
        axes[1].set_title("True Label")
        axes[1].set_axis_off()

        # 予測結果
        axes[2].imshow(predict[0].cpu().permute(1, 2, 0).numpy())
        axes[2].set_title("Prediction")

        pyplot.show()

    def _save_model(self, model: Module, model_name: str):
        self.model_root.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_root / f"{model_name}.pth")

    def _save_logs(self, log, file_name):
        file_path = self.log_root / file_name
        self.log_root.mkdir(parents=True, exist_ok=True)
        file_exists = file_path.exists() and file_path.stat().st_size > 0  # ファイルの存在と内容の有無を確認
        with file_path.open("a", newline="", encoding="utf-8") as file:
            header = list(log[0].keys())
            writer = DictWriter(file, fieldnames=header)
            if not file_exists:
                writer.writeheader()  # ファイルが空の場合にヘッダーを書き込む
            writer.writerows(log)

    def fit(self, model: Module, optimizer: Optimizer, epochs: int = 50):
        best_score = 0

        model.to(self.device)
        for epoch in range(epochs):
            print(f"epoch: {epoch:>4}/{epochs:<4}")
            print("test")
            print("-" * 100)
            train_report = self._train(epoch, model, optimizer)
            self._save_logs(train_report, "train.csv")
            print("test")
            print("-" * 100)
            test_report = self._test(epoch, model)
            self._save_logs(test_report, "test.csv")
            print("visualize")
            print("-" * 100)
            self._visualize(model, epoch)
            self._save_model(model, f"{self.model_name}_{epoch}")
            score = self.metrics_score(test_report)
            if score > best_score:
                best_score = score
                self._save_model(model, f"best_score_{best_score}_{self.model_name}_{epoch}")


def default_metric_score(metrics: list[Dict[str, float]]):
    score = sum([-metrics[key] if key != "loss" else metrics[key] for key in metrics.keys()])
    return score


def get_timestamp(date_format):
    return datetime.now().strftime(date_format)
