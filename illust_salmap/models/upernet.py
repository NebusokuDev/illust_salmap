from torch.nn import Module
from torchinfo import summary

from illust_salmap.models.ez_bench import benchmark


class UperNet(Module):
    def __init__(self):
        super().__init__()
        self.model = UperNet()

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = UperNet()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)