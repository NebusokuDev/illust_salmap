from io import BytesIO

import requests
from PIL.Image import Image
from pytorch_lightning import Callback, LightningModule, Trainer


class DiscordNotifier:
    def __init__(self, url, quiet=True):
        self.url = url
        self.quiet = quiet

    def send(self, message, *images):
        data = {"content": message}

        files = []
        for image in images:
            if isinstance(image, Image):
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


class DiscordNotifyCallback(Callback, DiscordNotifier):
    def __init__(self, url, quiet=True):
        super().__init__(url=url, quiet=quiet)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.send("## Training started!")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.send("## Training finished!")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.send("## Testing started!")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.send("## Testing finished!")
        metrics = trainer.callback_metrics
        message = "## Testing completed with the following metrics\n"
        for metric_name, metric_value in metrics.items():
            message += f"- **{metric_name}**: {metric_value}\n"
        self.send(message)
