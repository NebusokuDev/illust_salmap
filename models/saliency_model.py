from pytorch_lightning import LightningModule
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torchmetrics import KLDivergence, CosineSimilarity
from torchmetrics.wrappers import LambdaInputTransformer

from training.metrics import NormalizedScanpathSaliency, normalized


class SaliencyModel(LightningModule):
    def __init__(self, model: Module, criterion: Module = None, lr=0.0001):
        super().__init__()
        self.model = model
        self.criterion = criterion or MSELoss()
        self.lr = lr

        # metrics
        # transform_target, transform_predに必要な関数を作成して
        self.kl_div = LambdaInputTransformer(KLDivergence(), transform_pred=normalized, transform_target=normalized)
        self.nss = LambdaInputTransformer(NormalizedScanpathSaliency(), transform_pred=normalized,
                                          transform_target=normalized)
        self.sim = LambdaInputTransformer(CosineSimilarity(), transform_pred=normalized, transform_target=normalized)
        self.scc = LambdaInputTransformer(CosineSimilarity(), transform_pred=normalized, transform_target=normalized)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)
        loss = self.criterion(predict, ground_truth)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_kl_div", self.kl_div, prog_bar=True)
        # self.log("train_nss", self.nss, prog_bar=True)
        # self.log("train_sim", self.sim, prog_bar=True)
        self.log("train_scc", self.scc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_kl_div", self.kl_div, prog_bar=True)
        # self.log("val_nss", self.nss, prog_bar=True)
        # self.log("val_sim", self.sim, prog_bar=True)
        self.log("val_scc", self.scc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_kl_div", self.kl_div, prog_bar=True)
        # self.log("test_nss", self.nss, prog_bar=True)
        # self.log("test_sim", self.sim, prog_bar=True)
        self.log("test_scc", self.scc, prog_bar=True)
