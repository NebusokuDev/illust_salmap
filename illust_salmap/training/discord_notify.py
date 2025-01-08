import time
from argparse import Namespace
from io import BytesIO
from typing import Any, Dict, Optional, Union

import requests
from PIL.Image import Image
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.transforms import ToTensor

from illust_salmap.dataset.imp1k import Imp1kDataset
from illust_salmap.training.utils import generate_plot


class DiscordNotifier:
    def __init__(self, url, quiet=True):
        self.url = url
        self.quiet = quiet

    def send(self, message, *images):
        data = {"content": message}

        files = []
        for image in images:  # 画像が複数渡される可能性に対応
            if isinstance(image, Image):  # PILのImage型を確認
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                files.append(("file", ("image.png", buffer, "image/png")))

        response = requests.post(self.url, data=data, files=files)

        if self.quiet:
            return

        if response.status_code == 204:
            print("Notification sent successfully!")
        else:
            print(f"Failed to send notification. Status code: {response.status_code}")


class DiscordNotifyCallback(Callback):
    def __init__(self, url, quiet=True):
        self.notifier = DiscordNotifier(url, quiet)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.notifier.send("Training finished!")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.notifier.send("## Testing finished!")
        metrics = trainer.callback_metrics
        # metrics を使って通知メッセージを作成する例:
        message = "## Testing completed with the following metrics\n"
        for metric_name, metric_value in metrics.items():
            message += f"- **{metric_name}**: {metric_value}\n"
        self.notifier.send(message)