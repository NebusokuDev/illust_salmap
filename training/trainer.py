import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(self,
                 train_data: Dataset,
                 test_data,
                 criterion: Module,
                 device: torch.device,
                 batch_stride: int = 3,
                 metrics_functions: dict = None,
                 ):
        if metrics_functions is None:
            metrics_functions = []
        self.device = device
        self.batch_stride = batch_stride
        self.train_data = DataLoader(train_data, pin_memory=True, num_workers=8, shuffle=True)
        self.test_data = DataLoader(test_data, pin_memory=True, num_workers=8, shuffle=False)
        self.criterion = criterion
        self.metrics_functions = metrics_functions

    def train(self, model, optimizer):
        model.train()  # モデルをトレーニングモードに設定

        for batch_idx, (image, label) in enumerate(self.train_data):
            image, label = image.to(self.device), label.to(self.device)

            optimizer.zero_grad()
            predict = model(image)
            loss = self.criterion(predict, label)
            loss.backward()
            optimizer.step()
            if batch_idx % self.batch_stride == 0:
                self._metrics(predict, label, loss)

    def test(self, model):
        model.eval()

        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(self.test_data):
                image, label = image.to(self.device), label.to(self.device)

                predict = model(image)
                loss = self.criterion(predict, label)
                if batch_idx % self.batch_stride == 0:
                    self._metrics(predict, label, loss)

    def _metrics(self, predict: Tensor, label: Tensor, loss: Tensor):
        pass

    def _visualize(self, model: Module):
        pass

    def _save_model(self, model: Module):
        pass

    def fit(self, model, optimizer, epochs=50):
        for epoch in range(epochs):
            print(f"epoch: {epoch:>4}/{epochs:<4}")
            self.train(model, optimizer)

            print("test")
            self.test(model)

    def __call__(self, model: Module, optimizer: Optimizer, epochs: int = 50):
        self.fit(model, optimizer, epochs)
