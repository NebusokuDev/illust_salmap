from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import MSELoss, Module
from torch.optim import Adam
from torchmetrics import AUROC, CosineSimilarity, KLDivergence
from torchmetrics.image import SpatialCorrelationCoefficient

from illust_salmap.training.metrics import convert_auroc, convert_kl_div, convert_scc, convert_sim, normalized
from illust_salmap.training.utils import generate_plot


class SaliencyModel(LightningModule):
    def __init__(self, model: Module, criterion: Module = None, lr: float = 0.0001):
        super().__init__()
        self.model = model
        self.criterion = criterion or MSELoss()
        self.lr = lr

        # metrics
        self.train_kl_div = KLDivergence()
        self.train_sim = CosineSimilarity(reduction="mean")
        self.train_scc = SpatialCorrelationCoefficient()
        self.train_auroc = AUROC("binary")

        self.val_kl_div = KLDivergence()
        self.val_sim = CosineSimilarity(reduction="mean")
        self.val_scc = SpatialCorrelationCoefficient()
        self.val_auroc = AUROC("binary")

        self.test_kl_div = KLDivergence()
        self.test_sim = CosineSimilarity(reduction="mean")
        self.test_scc = SpatialCorrelationCoefficient()
        self.test_auroc = AUROC("binary")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        return {"loss": loss, "predict": predict}

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        loss = outputs["loss"]
        predict = outputs["predict"]
        image, ground_truth = batch

        kl_div_pred, kl_div_ground = convert_kl_div(predict, ground_truth)
        sim_pred, sim_ground = convert_sim(predict, ground_truth)
        scc_pred, scc_ground = convert_scc(predict, ground_truth)
        auroc_pred, auroc_ground = convert_auroc(predict, ground_truth)

        self.train_kl_div(kl_div_pred, kl_div_ground)
        self.train_sim(sim_pred, sim_ground)
        self.train_scc(scc_pred, scc_ground)
        self.train_auroc(auroc_pred, auroc_ground)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("train_kl_div", self.train_kl_div, on_step=False, on_epoch=True, enable_graph=False)
        self.log("train_sim", self.train_sim, on_step=False, on_epoch=True, enable_graph=False)
        self.log("train_scc", self.train_scc, on_step=False, on_epoch=True, enable_graph=False)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, enable_graph=False)

    def validation_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        return {"val_loss": loss, "val_predict": predict}

    @torch.no_grad()
    def on_validation_batch_end(
            self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss = outputs["val_loss"]
        predict = outputs["val_predict"]
        image, ground_truth = batch

        kl_div_pred, kl_div_ground = convert_kl_div(predict, )
        sim_pred, sim_ground = convert_sim(predict, ground_truth)
        scc_pred, scc_ground = convert_scc(predict, ground_truth)
        auroc_pred, auroc_ground = convert_auroc(predict, ground_truth)

        self.val_kl_div(kl_div_pred, kl_div_ground)
        self.val_sim(sim_pred, sim_ground)
        self.val_scc(scc_pred, scc_ground)
        self.val_auroc(auroc_pred, auroc_ground)

        self.log("val_loss", loss, on_step=False, on_epoch=True, enable_graph=False)
        self.log("val_kl_div", self.val_kl_div, on_step=False, on_epoch=True, enable_graph=False)
        self.log("val_sim", self.val_sim, on_step=False, on_epoch=True, enable_graph=False)
        self.log("val_scc", self.val_scc, on_step=False, on_epoch=True, enable_graph=False)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, enable_graph=False)

        if batch_idx == 0:
            self.save_image("validation", self.trainer.current_epoch, image, ground_truth, predict)

    def test_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        detached_pred = predict.detach().clone()
        detached_ground = ground_truth.detach().clone()

        del detached_pred
        del detached_ground
        torch.cuda.empty_cache()

        return {"test_loss": loss, "test_predict": predict}

    @torch.no_grad()
    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        loss = outputs["test_loss"]
        predict = outputs["test_predict"]
        image, ground_truth = batch

        kl_div_pred, kl_div_ground = convert_kl_div(predict, ground_truth)
        sim_pred, sim_ground = convert_sim(predict, ground_truth)
        scc_pred, scc_ground = convert_scc(predict, ground_truth)
        auroc_pred, auroc_ground = convert_auroc(predict, ground_truth)

        self.test_kl_div(kl_div_pred, kl_div_ground)
        self.test_sim(sim_pred, sim_ground)
        self.test_scc(scc_pred, scc_ground)
        self.test_auroc(auroc_pred, auroc_ground)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("test_kl_div", self.test_kl_div, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("test_sim", self.test_sim, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("test_scc", self.test_scc, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)

        if batch_idx == self.trainer.num_test_batches[0] - 1:
            self.save_image("test", self.trainer.current_epoch, image, ground_truth, predict)

    @torch.no_grad()
    def save_image(self, stage: str, epoch: int, images: Tensor, ground_truths: Tensor, predicts: Tensor) -> None:
        # 画像を正規化
        images = normalized(images)
        ground_truths = normalized(ground_truths)
        predicts = normalized(predicts)
        title = f"{stage}_images: {epoch}"

        plot = generate_plot(title, {"input": images[0], "ground_truth": ground_truths[0], "predict": predicts[0]})

        # TensorBoardに画像を追加
        self.logger.experiment.add_image(f"{stage}_images", plot, global_step=epoch)
