import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(self,
                 train_data: Dataset,
                 test_data,
                 criterion: Module,
                 device: torch.device,
                 batch_stride: int = 3):
        self.device = device
        self.batch_stride = batch_stride
        self.train_data = DataLoader(train_data, pin_memory=True, num_workers=8, shuffle=True)
        self.test_data = DataLoader(test_data, pin_memory=True, num_workers=8, shuffle=False)
        self.criterion = criterion

    def clear_metrics(self):
        pass

    def train(self, model, optimizer):
        model.train()  # モデルをトレーニングモードに設定

        for batch_idx, (image, salmap) in enumerate(self.train_data):
            image = image.to(self.device)
            salmap = salmap.to(self.device)

            optimizer.zero_grad()

            outputs = model(image)
            loss = self.criterion(outputs, salmap)
            loss.backward()
            optimizer.step()

    def test(self, model):
        model.eval()

        with torch.no_grad():
            for batch_idx, (image, salmap) in enumerate(self.test_data):
                image = image.to(self.device)
                salmap = salmap.to(self.device)

                outputs = model(image)
                loss = self.criterion(outputs, salmap)

    def fit(self, model, optimizer, epochs=50):
        for epoch in range(1, epochs + 1):
            print(f"epoch: {epoch:>4}/{epochs:<4}")
            self.train(model, optimizer)

            print("test")
            self.test(model)
