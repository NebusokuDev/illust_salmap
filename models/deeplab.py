from torchsummary import summary
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

if __name__ == '__main__':
    summary(deeplabv3_mobilenet_v3_large(), (3, 256, 256))
