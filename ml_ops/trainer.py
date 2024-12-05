import time

import torch


class Trainer():
    def __init__(self, criterion,train_dataloader, test_dataloader, show_stride=10, epochs=50, device=torch.device("cpu")):
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.show_stride = show_stride

    def _training(self):
        for batch_index in range(200):

            time.sleep(0.05)
            loss = torch.rand(1)
            if batch_index % self.show_stride == 0:
                print(f"batch: {batch_index: >5}",
                      f"loss: {loss.item():8.5f}",
                      sep="\t"
                      )

    def _testing(self):
        pass

    def save_model(self):
        time.sleep(0.5)

    def fit(self):
        for epoch in range(self.epochs):

            print(f"epoch:{epoch:>4}/{self.epochs:<3}")
            print("-" * 100)
            print("training")
            print("-"*100)
            self._training()
            print("testing")
            print("-" * 100)
            self._testing()
            self.save_model()
            print("-"* 100)

    def __call__(self, *args, **kwargs):
        self.fit()

