from pathlib import Path
import torch
import yaml
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, Optional


class Trainer:
    def __init__(self,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 criterion: Module,
                 device: torch.device,
                 batch_stride: int = 3,
                 metrics_functions: Optional[Dict[str, Callable[[Tensor, Tensor], float]]] = None,
                 log_dir: str = "./logs",
                 model_dir: str = "./",
                 date_format: str = "%Y/%m/%d-%H"):
        self.device = device
        self.batch_stride = batch_stride
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.metrics = metrics_functions or {}
        self.log_dir = Path(log_dir).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.date_format = date_format
        self.summary_writer = SummaryWriter()

    def _train(self, model: Module, optimizer: Optimizer):
        model.train()
        for batch_idx, (image, label) in enumerate(self.train_dataloader):
            image, label = image.to(self.device), label.to(self.device)
            optimizer.zero_grad()
            predict = model(image)
            loss = self.criterion(predict, label)
            loss.backward()
            optimizer.step()
            if batch_idx % self.batch_stride == 0:
                self._eval_metrics(predict, label, loss)

    def _test(self, model: Module):
        model.eval()
        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(self.test_dataloader):
                image, label = image.to(self.device), label.to(self.device)
                predict = model(image)
                loss = self.criterion(predict, label)
                if batch_idx % self.batch_stride == 0:
                    self._eval_metrics(predict, label, loss)

    def _eval_metrics(self, predict: Tensor, label: Tensor, loss: Tensor):
        self.summary_writer.add_scalar("loss", loss.item())
        for metric_label, metric_fn in self.metrics.items():
            metric = metric_fn(predict, label)
            self.summary_writer.add_scalar(metric_label, metric)

    def _visualize(self, model: Module):
        image, label = next(iter(self.test_dataloader))
        image, label = image.to(self.device), label.to(self.device)
        with torch.no_grad():
            predict = model(image)
        # Example visualization code
        from torchvision.utils import make_grid
        self.summary_writer.add_image("Predictions", make_grid(predict), global_step=0)

    def _save_log(self, log: dict, name: str):
        with (self.log_dir / name).open("w") as f:
            yaml.dump(log, f)

    def _save_model(self, model: Module, name: str):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_dir / f"{name}.pth")

    def fit(self, model: Module, optimizer: Optimizer, epochs: int = 50):
        model.to(self.device)
        for epoch in range(epochs):
            print(f"epoch: {epoch:>4}/{epochs:<4}")
            print("-" * 100)
            self._train(model, optimizer)
            self._test(model)
            self._visualize(model)
            self._save_model(model, f"{epoch}")

    def __call__(self, model: Module, optimizer: Optimizer, epochs: int = 50):
        self.fit(model, optimizer, epochs)
