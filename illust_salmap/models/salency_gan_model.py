import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam


class SaliencyGANModel(LightningModule):
    def __init__(self, generator: Module, discriminator: Module, criterion):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = criterion

        self.kl_div = None
        self.scc = None
        self.auroc = None

    def forward(self, x) -> Tensor:
        return self.generator(x)

    def configure_optimizers(self):
        optimizer_g = Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [optimizer_g, optimizer_d], []

    def generator_loss(self, predict, ground_truth, reality):
        reconstruct_loss = self.criterion(predict, ground_truth)
        adversarial_loss = self.criterion(reality, torch.ones())
        return reconstruct_loss * 0.5 + adversarial_loss * 0.5

    def discriminator_loss(self, real_predict, fake_predict):
        real_loss = self.criterion(real_predict, torch.ones())
        fake_loss = self.criterion(fake_predict, torch.zeros())
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
            real_pred = self.discriminator(image)
            predict_map = self.generator(image).detach()
            fake_pred = self.discriminator(predict_map)
            real_pred = self.discriminator(ground_truth)
            loss_d = self.discriminator_loss(real_pred, reality)
            self.log('train_loss_d', loss_d)
            return loss_d

    def validation_step(self, batch, batch_idx):
        real_images = batch[0]
        batch_size = real_images.size(0)

        # ランダムノイズベクトル
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)

        fake_images = self.generator(z)
        real_pred = self.discriminator(real_images)
        fake_pred = self.discriminator(fake_images)

        loss_d = self.discriminator_loss(real_pred, fake_pred)
        self.log('val_loss_d', loss_d)

        return loss_d
