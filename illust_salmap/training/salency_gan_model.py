import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam


class SaliencyGANModel(LightningModule):
    def __init__(self, generator: Module, discriminator: Module, criterion, lr=1e-4, latent_dim=100):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = criterion
        self.lr = lr
        self.latent_dim = latent_dim

    def forward(self, x) -> Tensor:
        return self.generator(x)

    def configure_optimizers(self):
        optimizer_g = Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [optimizer_g, optimizer_d], []

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

        # Generatorのトレーニング
        if optimizer_idx == 0:
            predict_map = self.generator(image)
            reality = self.discriminator(predict_map)
            loss_g = self.generator_loss(predict_map, ground_truth, reality)
            self.log('train_loss_g', loss_g)
            return loss_g

        # Discriminatorのトレーニング
        if optimizer_idx == 1:
            predict_map = self.generator(image)

            real_input = torch.cat((image, ground_truth), dim=1)
            fake_input = torch.cat((image, predict_map), dim=1)

            real_pred = self.discriminator(real_input)
            fake_pred = self.discriminator(fake_input)

            loss_d = self.discriminator_loss(real_pred, fake_pred)
            return loss_d

    def validation_step(self, batch, batch_idx):
        image, ground_truth = batch

        predict_map = self.generator(image)

        real_input = torch.cat((image, ground_truth), dim=1)
        fake_input = torch.cat((image, predict_map), dim=1)

        real_pred = self.discriminator(real_input)
        fake_pred = self.discriminator(fake_input)

        loss_d = self.discriminator_loss(real_pred, fake_pred)
        return loss_d