from pytorch_lightning import Trainer
from torch.nn import MSELoss

from dataset.cat2000 import Cat2000DataModule
from models.saliency_model import SaliencyModel
from models.unet import UNet

if __name__ == '__main__':
    datamodule = Cat2000DataModule()
    criterion = MSELoss()
    model = UNet()
    lit_model = SaliencyModel(model, MSELoss())
    trainer = Trainer()
    trainer.fit(lit_model, datamodule=datamodule)
