import os
import time

import numpy as np
import psutil
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FakeData
from torchvision.models.quantization import resnet18
from torchvision.transforms import ToTensor


@torch.no_grad()
def benchmark(model: Module, shape: tuple, device: str = "cpu", num_sample=5):
    model = model.to(device)
    dummy_input = torch.randn(shape, device=device)

    times = []
    memory_usages = []

    print("=" * 100)
    for i in range(num_sample):
        start_time = time.perf_counter()  # より高精度なタイマーを使用
        model(dummy_input)
        end_time = time.perf_counter()
        iteration_time = end_time - start_time
        times.append(iteration_time)
        memory_usages.append(get_memory_usage(device))
        print(f"Iteration {i + 1}/{num_sample} [time: {times[i]:.6f}s, memory usage: {display_byte_unit(memory_usages[i])}]")
        torch.cuda.empty_cache()

    avg_time = sum(times) / num_sample
    std_dev = np.std(times)
    avg_memory_usage = sum(memory_usages) / num_sample

    print("=" * 100)
    print("Benchmark results")
    print("=" * 100)
    print("Model:", type(model).__name__)
    print("Device:", device)
    print("Batch size:", shape[0])
    print("Input shape:", shape[1:])
    print(f"Average time: {avg_time:.6f}s")
    print(f"Standard deviation: {std_dev:.6f}s")
    print(f"Average memory usage: {display_byte_unit(avg_memory_usage)}")
    print("=" * 100)


def estimate_total_training_time(
        model: Module,
        dataset: Dataset,
        criterion: Module,
        batch_size: int = 32,
        optimizer_builder: callable = lambda params: Adam(params),
        epochs: int = 100,
        device="cpu",
        num_steps: int = 1,
        num_samples: int = 1, ):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = model.to(device)
    optimizer = optimizer_builder(model.parameters())

    times = []
    for _ in range(num_samples):
        times.append(measure_step_time(model, dataloader, criterion, optimizer, device=device, num_steps=num_steps))

    avg_time = sum(times) / num_samples
    epoch_time = avg_time * len(dataloader)
    total_time = epoch_time * epochs

    print("=" * 100)
    print("Estimated training time")
    print("=" * 100)
    print("Model:", type(model).__name__)
    print("Device:", device)
    print("Batch size:", batch_size)
    print("Input shape:", dataset[0][0].shape)
    print("Epochs:", epochs)
    print("Number of samples:", len(dataset))
    print(f"Estimated training epoch time: {display_time_unit(epoch_time)}")
    print(f"Estimated total training time for {epochs} epochs: {display_time_unit(total_time)}")
    print("=" * 100)


def measure_step_time(
        model: Module, dataloader: DataLoader, criterion: Module, optimizer: Optimizer, device, num_steps=5
):
    start_time = time.perf_counter()

    model.train()

    if num_steps <= 0:
        num_steps = len(dataloader)

    num_steps = min(num_steps, len(dataloader))

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_steps:
            break
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.perf_counter()

    return (end_time - start_time) / num_steps


def get_memory_usage(device: str = "cpu"):
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    elif device == "cpu":
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    else:
        raise ValueError("Unsupported device type. Use 'cpu' or 'cuda'.")


def display_time_unit(t: float):
    days = int(t // 86400)
    hours = int((t % 86400) // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)

    time_units = []
    if days > 0:
        time_units.append(f"{days}d")
    if hours > 0:
        time_units.append(f"{hours}h")
    if minutes > 0:
        time_units.append(f"{minutes}m")
    if seconds > 0 or len(time_units) == 0:
        time_units.append(f"{seconds}s")

    return ' '.join(time_units)


def display_byte_unit(usage: float):
    if usage < 1024:
        return f"{usage:.2f}B"
    elif usage < 1024 ** 2:
        return f"{usage / 1024:.2f}KB"
    elif usage < 1024 ** 3:
        return f"{usage / 1024 ** 2:.2f}MB"
    else:
        return f"{usage / 1024 ** 3:.2f}GB"


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (1, 3, 256, 256)
    model = resnet18()

    benchmark(model, (1, 3, 256, 256), device=device)
    estimate_total_training_time(model, FakeData(num_classes=1000, transform=ToTensor()), CrossEntropyLoss())
