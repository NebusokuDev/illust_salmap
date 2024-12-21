from pytorch_lightning import Trainer
from torch.nn import MSELoss

from dataset.cat2000 import Cat2000
from models.saliency_model import SaliencyModel
from models.unet_lite import UNetLite

if __name__ == '__main__':
    datamodule = Cat2000()
    print(datamodule)
    criterion = MSELoss()
    model = UNetLite()
    model.decoder_32_out.use_skip_connection = False
    lit_model = SaliencyModel(model, MSELoss())
    trainer = Trainer(
    )
    trainer.fit(
        lit_model,
        datamodule=datamodule
    )
