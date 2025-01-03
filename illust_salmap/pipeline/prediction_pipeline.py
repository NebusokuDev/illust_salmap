from pathlib import Path

import torch
from PIL.Image import Image
from torch.nn import Module
from torchvision.transforms.v2 import Transform


class PredictionPipeline:
    def __init__(
            self,
            model: Module,
            ckpt_path: Path,
            preprocess: Transform | callable,
            postprocess: Transform | callable, ):
        self.model = model.load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.preprocess = preprocess
        self.postprocess = postprocess

    @torch.no_grad()
    def prediction(self, images: list[Image]):
        batch = self.preprocess(images)

        predict = self.model(batch)

        result = self.postprocess(predict)

        del predict, batch
        torch.cuda.empty_cache()

        return result

if __name__ == '__main__':
    from torchvision.transforms.v2 import Compose, Resize, ToTensor
    from torchvision.transforms.v2.functional import to_pil_image
    from illust_salmap.models.unet_v2 import UNetV2

    path = input("path: ")

    postprocess = Compose([to_pil_image, Resize((256, 256)), ToTensor()])
    preprocess = Compose([Resize((256, 256)), ToTensor()])

    pipeline = PredictionPipeline(UNetV2(), Path(path), preprocess, postprocess)