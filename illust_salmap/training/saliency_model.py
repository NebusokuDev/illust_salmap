from typing import Any

import torch
from PIL import Image
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torchmetrics import KLDivergence, AUROC, CosineSimilarity
from torchmetrics.image import SpatialCorrelationCoefficient

from illust_salmap.training.metrics import convert_kl_div, normalized, convert_sim, convert_scc, convert_auroc
from illust_salmap.training.utils import generate_plot


class SaliencyModel(LightningModule):
    def __init__(self, model: Module, criterion: Module = None, lr: float = 0.0001):
        super().__init__()
        self.model = model
        self.criterion = criterion or MSELoss()
        self.lr = lr

        # metrics
        self.train_kl_div = KLDivergence()
        self.train_sim = CosineSimilarity()
        self.train_scc = SpatialCorrelationCoefficient()
        # self.train_cc = SpearmanCorrCoef()
        self.train_auroc = AUROC("binary")

        self.val_kl_div = KLDivergence()
        self.val_sim = CosineSimilarity()
        self.val_scc = SpatialCorrelationCoefficient()
        # self.val_cc = SpearmanCorrCoef()
        self.val_auroc = AUROC("binary")

        self.test_kl_div = KLDivergence()
        self.test_sim = CosineSimilarity()
        self.test_scc = SpatialCorrelationCoefficient()
        # self.test_cc = SpearmanCorrCoef()
        self.test_auroc = AUROC("binary")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        detached_pred = predict.detach().clone().cpu()
        detached_ground = ground_truth.detach().clone().cpu()

        kl_div_pred, kl_div_ground = convert_kl_div(detached_pred, detached_ground)
        sim_pred, sim_ground = convert_sim(detached_pred, detached_ground)
        scc_pred, scc_ground = convert_scc(detached_pred, detached_ground)
        auroc_pred, auroc_ground = convert_auroc(detached_pred, detached_ground)

        self.train_kl_div(kl_div_pred, kl_div_ground)
        self.train_sim(sim_pred, sim_ground)
        self.train_scc(scc_pred, scc_ground)
        self.train_auroc(auroc_pred, auroc_ground)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("train_kl_div", self.train_kl_div, on_step=False, on_epoch=True, enable_graph=False)
        self.log("train_sim", self.train_sim, on_step=False, on_epoch=True, enable_graph=False)
        self.log("train_scc", self.train_scc, on_step=False, on_epoch=True, enable_graph=False)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, enable_graph=False)

        return loss

    def validation_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        detached_pred = predict.detach().clone().cpu()
        detached_ground = ground_truth.detach().clone().cpu()

        kl_div_pred, kl_div_ground = convert_kl_div(detached_pred, detached_ground)
        sim_pred, sim_ground = convert_sim(detached_pred, detached_ground)
        scc_pred, scc_ground = convert_scc(detached_pred, detached_ground)
        auroc_pred, auroc_ground = convert_auroc(detached_pred, detached_ground)

        self.val_kl_div(kl_div_pred, kl_div_ground)
        self.val_sim(sim_pred, sim_ground)
        self.val_scc(scc_pred, scc_ground)
        self.val_auroc(auroc_pred, auroc_ground)

        self.log("val_loss", loss, on_step=False, on_epoch=True, enable_graph=False)
        self.log("val_kl_div", self.val_kl_div, on_step=False, on_epoch=True, enable_graph=False)
        self.log("val_sim", self.val_sim, on_step=False, on_epoch=True, enable_graph=False)
        self.log("val_scc", self.val_scc, on_step=False, on_epoch=True, enable_graph=False)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, enable_graph=False)

        return loss

    def test_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        detached_pred = predict.detach().clone()
        detached_ground = ground_truth.detach().clone()

        kl_div_pred, kl_div_ground = convert_kl_div(detached_pred, detached_ground)
        sim_pred, sim_ground = convert_sim(detached_pred, detached_ground)
        scc_pred, scc_ground = convert_scc(detached_pred, detached_ground)
        auroc_pred, auroc_ground = convert_auroc(detached_pred, detached_ground)

        self.test_kl_div(kl_div_pred, kl_div_ground)
        self.test_sim(sim_pred, sim_ground)
        self.test_scc(scc_pred, scc_ground)
        self.test_auroc(auroc_pred, auroc_ground)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("test_kl_div", self.test_kl_div, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("test_sim", self.test_sim, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("test_scc", self.test_scc, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)

        return loss

    @torch.no_grad()
    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            image, ground_truth = batch

            predict = self.forward(image)

            self.save_image("validation", self.trainer.current_epoch, image, ground_truth, predict)

    @torch.no_grad()
    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == self.trainer.num_test_batches[0] - 1:
            image, ground_truth = batch

            predict = self.forward(image)

            self.save_image("test", self.trainer.current_epoch, image, ground_truth, predict)

    @torch.no_grad()
    def save_image(self, stage: str, epoch: int, images: Tensor, ground_truths: Tensor, predicts: Tensor) -> None:
        # 画像を正規化
        images = normalized(images)
        ground_truths = normalized(ground_truths)
        predicts = normalized(predicts)

        plot = generate_plot({"input": images[0], "ground_truth": ground_truths[0], "predict": predicts[0]})

        Image.open(plot).save(f"./{stage}_{epoch}.png")

        # TensorBoardに画像を追加
        self.logger.experiment.add_image(f"{stage}_images", plot, global_step=epoch)
