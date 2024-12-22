from typing import Any

import torch
from matplotlib import pyplot
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, cuda
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torchmetrics import KLDivergence, AUROC, CosineSimilarity, SpearmanCorrCoef
from torchmetrics.image import SpatialCorrelationCoefficient

from training.metrics import convert_kl_div, normalized, convert_sim, convert_scc, convert_auroc


class SaliencyModel(LightningModule):
    """
    A PyTorch Lightning model for saliency prediction.

    This model predicts saliency maps from input images and computes loss and evaluation metrics
    based on the predicted saliency maps and the provided ground truth. The model also supports
    logging of training, validation, and test metrics, including loss, KL divergence, similarity,
    Spearman correlation coefficient (SCC), and AUROC. It also provides image visualizations
    after each training and test epoch.

    Attributes:
        model (Module): The underlying neural network model used for saliency prediction.
        criterion (Module): The loss function used for training. Defaults to MSELoss if not provided.
        lr (float): The learning rate for the optimizer.
        kl_div (callable): The KL divergence metric for evaluation.
        sim (callable): The similarity metric for evaluation.
        scc (callable): The Spearman correlation coefficient (SCC) metric for evaluation.
        auroc (callable): The area under the receiver operating characteristic (AUROC) metric for evaluation.

    Methods:
        forward(x): Performs a forward pass through the model.
        configure_optimizers(): Configures the optimizer (Adam) for the model.
        training_step(batch, batch_idx): Defines the training step, computes loss, and updates metrics.
        validation_step(batch, batch_idx): Defines the validation step, computes loss, and updates metrics.
        test_step(batch, batch_idx): Defines the test step, computes loss, and updates metrics.
        on_train_epoch_end(): Displays images at the end of the training epoch.
        on_test_epoch_end(): Displays images at the end of the test epoch.
        show_images(image, ground_truth, predict): Displays images, ground truth, and predictions in a grid.
    """

    def __init__(self, model: Module, criterion: Module = None, lr: float = 0.0001):
        """
        Initializes the SaliencyModel.

        Args:
            model (Module): The neural network model for saliency prediction.
            criterion (Module, optional): The loss function used for training. Defaults to MSELoss.
            lr (float, optional): The learning rate for the optimizer. Defaults to 0.0001.
        """
        super().__init__()
        self.model = model
        self.criterion = criterion or MSELoss()
        self.lr = lr

        # metrics
        self.kl_div = KLDivergence()
        self.sim = CosineSimilarity()
        self.scc = SpatialCorrelationCoefficient()
        self.cc = SpearmanCorrCoef()
        self.auroc = AUROC("binary")

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (Tensor): The input tensor for the model.

        Returns:
            Tensor: The predicted saliency map.
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            Adam: The Adam optimizer configured with the model's parameters and learning rate.
        """
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        Defines the training step, computes loss, and updates metrics.

        Args:
            batch (tuple): A tuple containing the input image and ground truth.
            batch_idx (int): The index of the current batch.

        Returns:
            Tensor: The computed loss for the batch.
        """
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        detached_pred = predict.detach().clone().cpu()
        detached_ground = ground_truth.detach().clone().cpu()

        kl_div_pred, kl_div_ground = convert_kl_div(detached_pred, detached_ground)
        sim_pred, sim_ground = convert_sim(detached_pred, detached_ground)
        scc_pred, scc_ground = convert_scc(detached_pred, detached_ground)
        auroc_pred, auroc_ground = convert_auroc(detached_pred, detached_ground)

        self.kl_div(kl_div_pred, kl_div_ground)
        self.sim(sim_pred, sim_ground)
        self.scc(scc_pred, scc_ground)
        self.auroc(auroc_pred, auroc_ground)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_kl_div", self.kl_div, prog_bar=True)
        self.log("train_sim", self.sim, prog_bar=True)
        self.log("train_scc", self.scc, prog_bar=True)
        self.log("train_auroc", self.auroc, prog_bar=True)

        del detached_pred, detached_ground
        cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step, computes loss, and updates metrics.

        Args:
            batch (tuple): A tuple containing the input image and ground truth.
            batch_idx (int): The index of the current batch.
        """
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        detached_pred = predict.detach().clone().cpu()
        detached_ground = ground_truth.detach().clone().cpu()

        kl_div_pred, kl_div_ground = convert_kl_div(detached_pred, detached_ground)
        sim_pred, sim_ground = convert_sim(detached_pred, detached_ground)
        scc_pred, scc_ground = convert_scc(detached_pred, detached_ground)
        auroc_pred, auroc_ground = convert_auroc(detached_pred, detached_ground)

        self.kl_div(kl_div_pred, kl_div_ground)
        self.sim(sim_pred, sim_ground)
        self.scc(scc_pred, scc_ground)
        self.auroc(auroc_pred, auroc_ground)

        self.log("val_loss", loss)
        self.log("val_kl_div", self.kl_div)
        self.log("val_sim", self.sim)
        self.log("val_scc", self.scc)
        self.log("val_auroc", self.auroc)

        del detached_pred, detached_ground
        cuda.empty_cache()

        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines the test step, computes loss, and updates metrics.

        Args:
            batch (tuple): A tuple containing the input image and ground truth.
            batch_idx (int): The index of the current batch.
        """
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        detached_pred = predict.detach().clone()
        detached_ground = ground_truth.detach().clone()

        kl_div_pred, kl_div_ground = convert_kl_div(detached_pred, detached_ground)
        sim_pred, sim_ground = convert_sim(detached_pred, detached_ground)
        scc_pred, scc_ground = convert_scc(detached_pred, detached_ground)
        auroc_pred, auroc_ground = convert_auroc(detached_pred, detached_ground)

        self.kl_div(kl_div_pred, kl_div_ground)
        self.sim(sim_pred, sim_ground)
        self.scc(scc_pred, scc_ground)
        self.auroc(auroc_pred, auroc_ground)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_kl_div", self.kl_div, prog_bar=True)
        self.log("test_sim", self.sim, prog_bar=True)
        self.log("test_scc", self.scc, prog_bar=True)
        self.log("test_auroc", self.auroc, prog_bar=True)

        del detached_pred, detached_ground
        cuda.empty_cache()

    @torch.no_grad()
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == self.trainer.num_training_batches - 1:
            image, ground_truth = batch

            predict = self.forward(image)

            self.show_images(f"training epoch {self.trainer.current_epoch}", image, ground_truth, predict)

            del predict
            cuda.empty_cache()

    @torch.no_grad()
    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            image, ground_truth = batch

            predict = self.forward(image)

            self.show_images(f"validation epoch {self.trainer.current_epoch}", image, ground_truth, predict)

            del predict
            cuda.empty_cache()

    @torch.no_grad()
    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == self.trainer.num_test_batches[0] - 1:
            image, ground_truth = batch

            predict = self.forward(image)

            self.show_images(f"test epoch {self.trainer.current_epoch}", image, ground_truth, predict)

            del predict
            cuda.empty_cache()

    @torch.no_grad()
    def show_images(self, title: str, images: Tensor, ground_truths: Tensor, predicts: Tensor) -> None:
        """
        Displays images, ground truth, and predictions in a grid.

        Args:
            images (Tensor): The input image.
            ground_truths (Tensor): The ground truth saliency map.
            predicts (Tensor): The predicted saliency map.
        """
        fig, axes = pyplot.subplots(1, 3, figsize=(11, 5))

        fig.suptitle(title)

        images = normalized(images)
        ground_truths = normalized(ground_truths)
        predicts = normalized(predicts)

        axes[0].set_title('input image')
        axes[0].imshow(images[0].cpu().permute(1, 2, 0).detach().numpy())
        axes[0].axis("off")

        axes[1].set_title('ground truth')
        axes[1].imshow(ground_truths[0].cpu().permute(1, 2, 0).detach().numpy())
        axes[1].axis("off")

        axes[2].set_title('predict')
        axes[2].imshow(predicts[0].cpu().permute(1, 2, 0).detach().numpy())
        axes[2].axis("off")

        pyplot.show()
        pyplot.close()
