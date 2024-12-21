import os

import matplotlib.pyplot as plt


def train(model, dataloader, criterion, optimizer, device, batch_stride=3):
    model.train()  # モデルをトレーニングモードに設定
    training_report = []
    total_loss = 0
    max_loss = float("-inf")
    min_loss = float("inf")

    for batch_idx, (image, salmap) in enumerate(dataloader):
        image = image.to(device)
        salmap = salmap.to(device)

        optimizer.zero_grad()

        outputs = model(image)
        loss = criterion(outputs, salmap)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        accuracy = eval_pixel_wise_accuracy(outputs.detach().clone().reshape(-1), salmap.detach().clone().reshape(-1))

        avg_loss = total_loss / (batch_idx + 1)

        if loss.item() > max_loss:
            max_loss = loss.item()
        if loss.item() < min_loss:
            min_loss = loss.item()

        error_range = max_loss - min_loss
        jaccard_index = eval_jaccard_index(outputs.detach().clone(), salmap.detach().clone())
        accuracy = eval_pixel_wise_accuracy(outputs.detach().clone(), salmap.detach().clone())

        training_report.append({
            "batch": batch_idx,
            "loss": loss.item(),
            "avg_loss": avg_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "pixel_wise_accuracy:": accuracy.item(),
            "error_range": error_range,
            "jaccard_index": jaccard_index.item(),
        })

        if batch_idx % batch_stride == 0:
            print(f"batch: {batch_idx:>5}/{len(dataloader):<5} ({batch_idx / len(dataloader):5.2%})",
                  f"loss: {loss.item():>9.4f}",
                  f"avg loss: {avg_loss:>9.4f}",
                  f"error range: {error_range:>9.4f}",
                  f"image: [min: {salmap.min():>9.4f},\t max:{salmap.max():>8.4f}]",
                  f"output: [min: {outputs.min():>9.4f},\t max:{outputs.max():>8.4f}]",
                  f"Pixel-wise accuracy: {accuracy.item():>5.2%}",
                  f"Jaccard index: {jaccard_index.item():>5.2%}",
                  sep="\t")

    average_loss = total_loss / len(dataloader)
    print(f"average loss: {average_loss:>9.4f}")
    print(f"max loss: {max_loss:>9.4f} min loss: {min_loss:>9.4f}")

    print()

    return training_report  # [{"batch":batch_idx, "loss":loss, "accuracy":accuracy}]


def validation(model, dataloader, criterion, device, batch_stride=2):
    model.eval()
    total_loss = 0
    total_jaccard_index = []
    total_accuracy = []

    with torch.no_grad():
        for batch_idx, (image, salmap) in enumerate(dataloader):
            image = image.to(device)
            salmap = salmap.to(device)

            outputs = model(image)
            loss = criterion(outputs, salmap)
            total_loss += loss.item()

            accuracy = eval_pixel_wise_accuracy(outputs.detach().clone().reshape(-1),
                                                salmap.detach().clone().reshape(-1))
            jaccard_index = eval_jaccard_index(outputs.detach().clone(), salmap.detach().clone())

            total_accuracy.append(accuracy.item())
            total_jaccard_index.append(jaccard_index.item())

            if batch_idx % batch_stride == 0:
                avg_loss = total_loss / (batch_idx + 1)

                print(f"batch: {batch_idx:>5}/{len(dataloader):<5} ({batch_idx / len(dataloader):>5.2%})",
                      f"loss: {loss:9.4f}",
                      f"avg loss: {avg_loss:9.4f}",
                      f"pixel wise accuracy: {accuracy:>5.2%}",
                      f"jaccard index {jaccard_index:>5.2%}",
                      sep="\t")

    avg_jaccard_index = sum(total_jaccard_index) / len(total_jaccard_index)
    avg_accuracy = sum(total_accuracy) / len(total_accuracy)
    average_loss = total_loss / len(dataloader)

    print(f"average loss: {average_loss:>9.4f}")
    print(f"average jaccard index: {avg_jaccard_index:>5.2%}")
    print(f"average accuracy: {avg_accuracy:>5.2%}")
    return average_loss


def fit(model, train_dataloader, test_dataloader, criterion, optimizer, epochs, device, root):
    """
    トレーニングを実行する関数
    """
    timestamp = get_timestamp()  # タイムスタンプを内部で取得
    best_loss = float("inf")
    best_accuracy = 0
    train_report = []  # [epoch, loss, ]

    for epoch in range(1, epochs + 1):
        print("-" * 60)
        print(f"epoch: {epoch:>4}/{epochs:<4}")
        print("-" * 60)
        print("train")
        print("-" * 60)
        epoch_report = train(model, train_dataloader, criterion, optimizer, device)
        print("test")
        print("-" * 60)
        test_loss = validation(model, test_dataloader, criterion, device)

        # エポックレポートにepoch番号を追加
        for report in epoch_report:
            report["epoch"] = epoch
        train_report.extend(epoch_report)

        # 最小の損失が出た場合はモデルを保存
        if epoch_report[-1]["loss"] < best_loss:
            best_loss = epoch_report[-1]["loss"]
            save_model(model, f"{root}/{timestamp}/best_loss", f"best_min_loss_{epoch}")

        # 2エポックごとにモデルを保存
        if epoch % 2 == 0:
            print("-" * 60)
            save_model(model, f"{root}/{timestamp}", f"{epoch}")

        # テストデータで結果を可視化
        visualize_batch_results(model, test_dataloader, device, )

        # CUDAメモリのクリア
        if device == "cuda":
            torch.cuda.empty_cache()

    print("-" * 60)
    print("training finished")
    print("-" * 60)


import torch
from os import path, makedirs
from datetime import datetime


def save_model(model, root, model_name):
    if not path.exists(root):
        makedirs(root)

    model_path = f"{root}/{model_name}.pth"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")


def get_timestamp(format="%Y/%m/%d/%H"):
    return datetime.now().strftime(format)


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
            makedirs(save_dir, exist_ok=True)

        # バッチ内の各画像を処理
        for i in range(5):
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def normalize01(tensor):
    return (tensor + 1) / 2


def eval_jaccard_index(saliency_map, ground_truth, threshold=0.05):
    saliency_map = normalize01(saliency_map)
    ground_truth = normalize01(ground_truth)
    intersection = torch.logical_and(saliency_map >= threshold, ground_truth >= threshold).sum()
    union = torch.logical_or(saliency_map >= threshold, ground_truth >= threshold).sum()
    return intersection / union


def eval_pixel_wise_accuracy(saliency_map, ground_truth, threshold=0.1):
    saliency_map = normalize01(saliency_map)
    ground_truth = normalize01(ground_truth)
    pixel_errors = torch.abs(saliency_map - ground_truth)

    correct_pixels = pixel_errors <= threshold

    accuracy = correct_pixels.float().mean()

    return accuracy
