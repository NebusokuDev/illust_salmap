from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torchmetrics import JaccardIndex


class SaliencyModel(LightningModule):
    def __init__(self, model: Module, criterion: Module = None, lr=0.0001):
        super().__init__()
        self.model = model
        self.criterion = criterion or MSELoss()
        self.train_accuracy = JaccardIndex(num_classes=10, task="multiclass")
        self.val_accuracy = JaccardIndex(num_classes=10, task="multiclass")
        self.test_accuracy = JaccardIndex(num_classes=10, task="multiclass")
        self.lr = lr
        self.validation_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)
        acc = self.train_accuracy(predict, label)

        print()
        print(image.shape)
        print(label.shape)
        print(predict.shape)

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

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        acc = self.test_accuracy(predict, label)

        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_acc": acc}
