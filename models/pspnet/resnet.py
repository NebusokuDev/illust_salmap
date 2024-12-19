from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential
from torchinfo import summary
from torchvision.models import resnet50, ResNet50_Weights


class ResNet(Module):
    def __init__(self):
        super().__init__()
        self.block1 = Sequential(
            Conv2d(3, 64, 2, 3, 1, bias=False),
            BatchNorm2d(64),
            ReLU()
        )

        self.block2 = Sequential(
            Conv2d(64, 64, 1, 3, 1, bias=False),
            BatchNorm2d(64),
            ReLU()
        )

        self.block3 = Sequential(
            Conv2d(64, 128, 1, 3, 1, bias=False),
            BatchNorm2d(128)
        )
        self.maxpool = MaxPool2d(3, 2, 1)

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avg_pool = resnet.avgpool
        self.fc = resnet.fc

        self.layer1[0].conv1 = Conv2d(128, 64, 1, 1, bias=False)
        self.layer1[0].downsample[0] = Conv2d(128, 256, 1, 1, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    summary(ResNet(), (4, 3, 256, 256))
