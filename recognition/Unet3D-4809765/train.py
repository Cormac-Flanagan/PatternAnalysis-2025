from dataset import NiiPairDataset
from modules import Unet3D, Diceloss
import torch.nn as nn
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
import os

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


def train(test_data, epochs=3):
    model = Unet3D(num_classes=6)
    model = model.to(device)
    criterion = Diceloss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    epochs_ = tqdm(range(epochs), total=epochs, desc="Epochs: ", leave=True)
    for epoch in epochs_:
        total_loss = 0.0
        for x, y in tqdm(test_data):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device_name):
                logits = model(x)
                loss = criterion(logits, y)
            with torch.autograd.set_detect_anomaly(True):
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        epochs_.set_postfix({"loss:": f"{total_loss:.4f}"})
        with open("logs.txt", "a") as f:
            f.write(f"{total_loss:.6f}\n")


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    dir = "./data"
    dataset = NiiPairDataset(dir)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    test_set, val_set, test_size = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    x, y = test_set[0]  # example sample
    x = x.to(device).unsqueeze(0)  # add batch dim
    y = y.to(device).unsqueeze(0)

    model = Unet3D(num_classes=6)
    model = model.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device_name):
        logits = model(x)
        loss = criterion(logits, y)

    peak = torch.cuda.max_memory_allocated(device)

    free_vram = torch.cuda.get_device_properties(device).total_memory - (
        torch.cuda.memory_reserved(device) + torch.cuda.memory_allocated(device)
    )

    safe_batch = max(1, np.floor(0.9 * free_vram / peak))
    print(safe_batch)

    loader = DataLoader(
        test_set,
        batch_size=safe_batch,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(8, os.cpu_count()),  # max 8 workers or CPU cores
    )
    train(loader)
