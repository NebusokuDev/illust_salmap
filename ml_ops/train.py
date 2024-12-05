import torch

from evaluation_function.eval import eval_pixel_wise_accuracy, eval_jaccard_index
from utils.utils import get_timestamp, visualize_batch_results, save_model


def train(model, dataloader, criterion, optimizer, device, batch_stride=3):
    model.train()  # モデルをトレーニングモードに設定
    training_report = []
    total_loss = 0
    max_loss = float("-inf")
    min_loss = float("inf")

    for batch_idx ,(image, salmap) in enumerate(dataloader):
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
            print(f"batch: {batch_idx:>5}/{len(dataloader):<5} ({batch_idx/len(dataloader):5.2%})",
                  f"loss: {loss.item():>9.4f}",
                  f"avg loss: {avg_loss:>9.4f}",
                  f"error range: {error_range:>9.4f}",
                  f"output: [min: {outputs.min():>9.4f},\t max:{outputs.max():>8.4f}]",
                  f"Pixel-wise accuracy: {accuracy.item():>5.2%}",
                  f"Jaccard index: {jaccard_index.item():>5.2%}",
                  sep="\t")

    average_loss = total_loss / len(dataloader)
    print(f"average loss: {average_loss:>9.4f}")
    print(f"max loss: {max_loss:>9.4f} min loss: {min_loss:>9.4f}")

    print()

    return training_report # [{"batch":batch_idx, "loss":loss, "accuracy":accuracy}]

def test(model, dataloader, criterion, device, batch_stride=2):
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

            accuracy = eval_pixel_wise_accuracy(outputs.detach().clone().reshape(-1), salmap.detach().clone().reshape(-1))
            jaccard_index = eval_jaccard_index(outputs.detach().clone(), salmap.detach().clone())

            total_accuracy.append(accuracy.item())
            total_jaccard_index.append(jaccard_index.item())

            if batch_idx % batch_stride == 0:
                avg_loss = total_loss / (batch_idx + 1)

                print(f"batch: {batch_idx:>5}/{len(dataloader):<5} ({batch_idx/len(dataloader):>5.2%})",
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
        test_loss = test(model, test_dataloader, criterion, device)

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