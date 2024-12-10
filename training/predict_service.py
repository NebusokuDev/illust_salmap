import torch
from torch.nn import Module


class PredictService:
    def __init__(self, model: Module, transform):
        self.transform = transform
        self.model = model

    def predict(self, image):
        with torch.no_grad:
            image = self.transform(image)
            saliency_map = self.model(image)
            return saliency_map

    def __call__(self, image):
        return self.predict(image)
