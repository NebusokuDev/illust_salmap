from PIL import Image
import torch
from torch.nn import Module
from torch import Tensor
from typing import Callable


class PredictService:
    def __init__(self, model: Module,
                 input_transform: Callable[[Image], Tensor],
                 output_transform: Callable[[Tensor], Image],
                 device: str
                 ):
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.model = model.to(device)
        self.device = device

    def predict(self, image: Image) -> Image:
        try:
            with torch.no_grad():
                # 画像の前処理 (PIL.Image -> Tensor)
                image_tensor = self.input_transform(image).to(self.device)

                # 予測
                predict: Tensor = self.model(image_tensor)
                predict = predict.detach().cpu()

                # 画像に後処理 (Tensor -> PIL.Image)
                return self.output_transform(predict)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
