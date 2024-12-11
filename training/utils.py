import os
from pathlib import Path

import torch
from os import path, makedirs
from datetime import datetime

from matplotlib import pyplot as plt


def save_model(model, root, model_name):
    save_path = Path(root / model_name)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model, save_path.with_suffix(".pth"))
    print(f"Model saved to {save_path}")


def visualize_batch_results(model, dataloader, device, figsize=(15, 5), save_dir=None):
    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():
        image, salmap = next(iter(dataloader))

        # モデルの出力を取得
        output = model(image.to(device)).detach().cpu()
        output = (output + 1) / 2
        image = (image + 1) / 2
        salmap = (salmap + 1) / 2

        # 保存ディレクトリが指定されている場合は作成
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # バッチ内の各画像を処理
        for i in range(len(image)):
            plt.figure(figsize=figsize)

            # 入力画像
            plt.subplot(1, 3, 1)
            plt.imshow(image[i].permute(1, 2, 0))
            plt.title("input")
            plt.axis("off")

            # ターゲット（サリエンシーマップ）
            plt.subplot(1, 3, 2)
            plt.imshow(salmap[i].permute(1, 2, 0), "gray")
            plt.title("target")
            plt.axis("off")

            # 出力結果
            plt.subplot(1, 3, 3)
            plt.imshow(output[i].permute(1, 2, 0), "gray")
            plt.title("output")
            plt.axis("off")

            # 保存または表示
            if save_dir:
                save_path = os.path.join(save_dir, f"result_{i}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Result {i} saved to {save_path}")

            plt.show()
            plt.close()


def choose_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda")

    return torch.device("cpu")
