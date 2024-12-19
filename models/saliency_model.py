from matplotlib import pyplot
from pytorch_lightning import LightningModule
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torchvision.utils import make_grid

from training.metrics import build_kl_div, build_sim, build_scc, build_auroc


class SaliencyModel(LightningModule):
    def __init__(self, model: Module, criterion: Module = None, lr=0.0001):
        super().__init__()
        self.model = model
        self.criterion = criterion or MSELoss()
        self.lr = lr

        # metrics
        self.kl_div = build_kl_div()
        self.sim = build_sim()
        self.scc = build_scc()
        self.auroc = build_auroc()

        self.validation_image_cache = []
        self.test_image_cache = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)
        loss = self.criterion(predict, ground_truth)

        # Update metrics
        self.kl_div(predict, ground_truth)
        self.sim(predict, ground_truth)
        self.scc(predict, ground_truth)
        self.auroc(predict, ground_truth)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_kl_div", self.kl_div, prog_bar=True)
        self.log("train_sim", self.sim, prog_bar=True)
        self.log("train_scc", self.scc, prog_bar=True)
        self.log("train_auroc", self.auroc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        # Update metrics
        self.kl_div(predict, ground_truth)
        self.sim(predict, ground_truth)
        self.scc(predict, ground_truth)
        self.auroc(predict, ground_truth)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_kl_div", self.kl_div, prog_bar=True)
        self.log("val_sim", self.sim, prog_bar=True)
        self.log("val_scc", self.scc, prog_bar=True)
        self.log("val_auroc", self.auroc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        self.kl_div(predict, ground_truth)
        self.sim(predict, ground_truth)
        self.scc(predict, ground_truth)
        self.auroc(predict, ground_truth)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_kl_div", self.kl_div, prog_bar=True)
        self.log("test_sim", self.sim, prog_bar=True)
        self.log("test_scc", self.scc, prog_bar=True)
        self.log("test_auroc", self.auroc, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        batch = next(iter(self.train_dataloader()))
        image, ground_truth, predict = batch

        self.show_images(image, ground_truth, predict)

        self.validation_image_cache.clear()

    def on_test_epoch_end(self) -> None:
        batch = next(iter(self.test_dataloader()))
        image, ground_truth, predict = batch

        self.show_images(image, ground_truth, predict)

        self.test_image_cache.clear()

    def show_images(self, image, ground_truth, predict) -> None:
        # 画像をグリッド形式に変換
        image_grid = make_grid(image[:6], nrow=3, padding=1, normalize=True)
        ground_truth_grid = make_grid(ground_truth[:6], nrow=3, padding=1, normalize=True)
        predict_grid = make_grid(predict[:6], nrow=3, padding=1, normalize=True)

        # 画像を表示する
        fig, axes = pyplot.subplots(1, 3, figsize=(16, 27))

        axes[0].set_title('input image')
        axes[0].imshow(image_grid.permute(1, 2, 0).cpu())
        axes[0].axis("off")

        axes[1].set_title('ground truth')
        axes[1].imshow(ground_truth_grid.permute(1, 2, 0).cpu())
        axes[1].axis("off")

        axes[2].set_title('predict')
        axes[2].imshow(predict_grid.permute(1, 2, 0).cpu())
        axes[2].axis("off")

        pyplot.show()
        pyplot.close()
