import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import Adam

from illust_salmap.training.metrics import normalized
from illust_salmap.training.utils import generate_plot


class SaliencyGANModel(LightningModule):
    def __init__(
            self,
            generator: Module,
            discriminator: Module,
            criterion=BCEWithLogitsLoss(),
            optimization_builder: callable = lambda params_g, params_d: (
                    Adam(params_g, lr=0.0001), Adam(params_d, lr=0.0001),), ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = criterion
        self.optimization_builder = optimization_builder

    def forward(self, x) -> Tensor:
        return self.generator(x)

    def configure_optimizers(self):
        return self.optimization_builder(self.generator.parameters(), self.discriminator.parameters())

    def generator_loss(self, predict, ground_truth, reality):
        reconstruct_loss = self.criterion(predict, ground_truth)
        adversarial_loss = self.criterion(reality, torch.ones_like(reality))
        return reconstruct_loss * 0.5 + adversarial_loss * 0.5

    def discriminator_loss(self, real_predict, fake_predict):
        real_loss = self.criterion(real_predict, torch.ones_like(real_predict))
        fake_loss = self.criterion(fake_predict, torch.zeros_like(fake_predict))
        return real_loss * 0.5 + fake_loss * 0.5

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, ground_truth = batch

        if optimizer_idx == 0:
            self.generator_training_step(image, ground_truth, batch_idx)
        else:
            self.discriminator_training_step(image, ground_truth, batch_idx)

    def generator_training_step(self, image, ground_truth, batch_idx):
        pass

    def discriminator_training_step(self, image, ground_truth, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        image, ground_truth = batch

        predict_map = self.generator(image)

        real_input = torch.cat((image, ground_truth), dim=1)
        fake_input = torch.cat((image, predict_map), dim=1)

        real_pred = self.discriminator(real_input)
        fake_pred = self.discriminator(fake_input)

        loss_d = self.discriminator_loss(real_pred, fake_pred)
        return loss_d

    @torch.no_grad()
    def save_image(self, stage: str, epoch: int, images: Tensor, ground_truths: Tensor, predicts: Tensor) -> None:
        images = normalized(images)
        ground_truths = normalized(ground_truths)
        predicts = normalized(predicts)
        title = f"{stage}_images: {epoch}"

        plot = generate_plot(title, {"input": images[0], "ground_truth": ground_truths[0], "predict": predicts[0]})

        self.logger.experiment.add_image(f"{stage}_images", plot, global_step=epoch)
