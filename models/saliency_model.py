import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torchmetrics import Accuracy, JaccardIndex


class SaliencyModel(LightningModule):
    def __init__(self, model: Module, criterion: Module, lr=1e-3):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.accuracy = Accuracy(task="binary")  # 精度計算のためにAccuracyを追加
        self.jaccard = JaccardIndex(task="binary")


    def forward(self, x) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, ground_truth = batch
        predict = self(image)
        loss = self.criterion(predict, ground_truth)
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, ground_truth = batch
        predict = self(image)
        loss = self.criterion(predict, ground_truth)

        # 精度を計算（任意で他の指標も計算できます）
        acc = self.accuracy(predict, ground_truth)

        # 結果として損失と精度を返します
        return {"test_loss": loss, "test_accuracy": acc}

    def test_epoch_end(self, outputs) -> STEP_OUTPUT:
        # test_stepの出力を集めて、最終的な結果を集計します
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()

        # 結果を返します
        return {"test_loss": avg_loss, "test_accuracy": avg_acc}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return AdamW(self.parameters(), lr=self.lr)
