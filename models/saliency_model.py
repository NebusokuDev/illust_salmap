from pytorch_lightning import LightningModule
from torch.nn import Module, MSELoss
from torch.optim import Adam

from training.metrics import build_kl_div, build_sim, build_scc, build_auroc


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
        validation_image_cache (list): A cache to store validation images for visualization.
        test_image_cache (list): A cache to store test images for visualization.

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
        self.kl_div = build_kl_div()
        self.sim = build_sim()
        self.scc = build_scc()
        self.auroc = build_auroc()

        self.validation_image_cache = []
        self.test_image_cache = []

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

        # # Update metrics
        # self.kl_div(predict, ground_truth)
        # self.sim(predict, ground_truth)
        # self.scc(predict, ground_truth)
        # self.auroc(predict, ground_truth)

        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_kl_div", self.kl_div, prog_bar=True)
        # self.log("train_sim", self.sim, prog_bar=True)
        # self.log("train_scc", self.scc, prog_bar=True)
        # self.log("train_auroc", self.auroc, prog_bar=True)

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

        # Update metrics
        # self.kl_div(predict.detach(), ground_truth.detach())
        # self.sim(predict.detach(), ground_truth.detach())
        # self.scc(predict.detach(), ground_truth.detach())
        # self.auroc(predict.detach(), ground_truth.detach())

        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_kl_div", self.kl_div, prog_bar=True)
        # self.log("val_sim", self.sim, prog_bar=True)
        # self.log("val_scc", self.scc, prog_bar=True)
        # self.log("val_auroc", self.auroc, prog_bar=True)

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

        self.kl_div(predict, ground_truth)
        self.sim(predict, ground_truth)
        self.scc(predict, ground_truth)
        self.auroc(predict, ground_truth)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_kl_div", self.kl_div, prog_bar=True)
        self.log("test_sim", self.sim, prog_bar=True)
        self.log("test_scc", self.scc, prog_bar=True)
        self.log("test_auroc", self.auroc, prog_bar=True)
