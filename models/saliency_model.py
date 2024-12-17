from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Module, MSELoss
from torch.optim import Adam


class SaliencyModel(LightningModule):
    def __init__(self, model: Module, criterion: Module = None, lr=0.0001):
        super().__init__()
        self.model = model
        self.criterion = criterion or MSELoss()
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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)
        return {"test_loss": loss}
