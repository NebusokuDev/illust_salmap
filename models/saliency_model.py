import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torchmetrics import JaccardIndex
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


class SaliencyModel(LightningModule):
    def __init__(self, model: Module, criterion: Module = None, lr=0.0001):
        super().__init__()
        self.model = model
        self.criterion = criterion or MSELoss()
        self.train_accuracy = JaccardIndex(num_classes=10, task="multiclass")
        self.val_accuracy = JaccardIndex(num_classes=10, task="multiclass")
        self.test_accuracy = JaccardIndex(num_classes=10, task="multiclass")
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)
        acc = self.train_accuracy(predict, label)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)
        acc = self.val_accuracy(predict, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "image": image, "label": label, "predict": predict}

    def validation_epoch_end(self, outputs):
        # バッチから必要なデータを収集
        x = torch.cat([output["x"] for output in outputs], dim=0)
        preds = torch.cat([output["preds"] for output in outputs], dim=0)
        y = torch.cat([output["y"] for output in outputs], dim=0)

        # 画像を可視化
        self.display_images(x, preds, y)

    def display_images(self, x, preds, y):
        # 入力画像、予測、ラベルをグリッド形式で表示
        grid_x = make_grid(x, nrow=4, normalize=True, range=(0, 1))
        grid_preds = make_grid(preds, nrow=4, normalize=True, range=(0, 1))
        grid_y = make_grid(y, nrow=4, normalize=True, range=(0, 1))

        # Matplotlibで描画
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        plt.title("Input")
        plt.imshow(grid_x.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Predictions")
        plt.imshow(grid_preds.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(grid_y.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        plt.show()
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        acc = self.test_accuracy(predict, label)

        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_acc": acc}
