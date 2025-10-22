from dataset import NiiPairDataset
from modules import Unet3D, Diceloss
import torch.nn as nn
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
import os, threading, queue
from utils import plot_results

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

def background_logger(q, path="./output/results.csv"):
    with open(path, "a") as f:
        while True:
            record = q.get()
            if record is None:
                break
            f.write(record + '\n')
            f.flush()

def test(model, loader):
    model.eval()
    criterion = Diceloss()
    criterion.to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device_name):
            loss = []
            for x, y in tqdm(loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss.append(criterion(logits, y).item())
    print(f"Max Dice Coefficient: {-1*min(loss):.4f}")
    print(f"Min Dice Coefficient: {-1*max(loss):.4f}")


def train(model, train_data, val_data, log_queue, epochs=10, val_rate=5, transforms=None):
    criterion = Diceloss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.grad_scaler.GradScaler()

    epochs_ = tqdm(range(epochs), total=epochs, desc="Epochs: ", leave=True)
    for epoch in epochs_:
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_data, leave=False, desc="Training"):
            x, y = x.to(device), y.to(device)
            if transforms is not None:
                x, y = transforms(x, y)
            optimizer.zero_grad()
            with torch.autocast(device_type=device_name):
                logits = model(x)
                loss = criterion(logits, y)
            with torch.autograd.set_detect_anomaly(True):
                scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        epochs_.set_postfix({"loss:": f"{total_loss:.4f}"})

        if epoch % val_rate == 0:
            del x, y
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in tqdm(val_data, leave=False, desc="Validation"):
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_name):
                        logits = model(x)
                        loss = criterion(logits, y)
                    val_loss += loss.item()
                log_queue.put(
                        f"{epoch}, {total_loss/len(train_data):.6f}, {val_loss/len(val_data):.6f}"
                    )
        else:
            log_queue.put(f"{epoch}, {total_loss/len(train_data):.6f}")


if __name__ == "__main__":
    aug_list = AugmentationSequential(
        K.RandomAffine3D(
            degrees=45,  # random rotations up to ±45°
            scale=(0.8, 1.2),  # random uniform scaling between 0.8× and 1.2×
            p=0.7,
        ),
        K.RandomHorizontalFlip3D(p=0.4),
        K.RandomVerticalFlip3D(p=0.4),
        data_keys=["input", "label"],
        same_on_batch=False,
    )

    torch.backends.cudnn.allow_tf32 = True
    dir = "./data"
    dataset = NiiPairDataset(dir, early_stop=0)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    x, y = train_set[0]  # example sample
    x = x.to(device).unsqueeze(0)  # add batch dim
    y = y.to(device).unsqueeze(0)

    model_ = Unet3D(num_classes=6)
    model_ = model_.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device_name):
        logits = model_(x)

    peak = torch.cuda.max_memory_allocated(device)

    free_vram = torch.cuda.get_device_properties(device).total_memory - (
        torch.cuda.memory_reserved(device) + torch.cuda.memory_allocated(device)
    )

    safe_batch = int(max(1, np.floor(0.9 * free_vram / peak)))
    print(safe_batch)

    loader = DataLoader(
        train_set,
        batch_size=safe_batch,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(8, os.cpu_count()),  # max 8 workers or CPU cores
    )

    val = DataLoader(
        val_set,
        batch_size=safe_batch,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(8, os.cpu_count()),  # max 8 workers or CPU cores
    )

    test_loader = DataLoader(
        test_set,
        batch_size=safe_batch,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(8, os.cpu_count()),  # max 8 workers or CPU cores
    )

    log_queue = queue.Queue()
    log_thread = threading.Thread(target=background_logger, args=(log_queue, ), daemon=True)
    log_thread.start()

    train(model_, loader, val, log_queue, epochs=30, transforms=aug_list)
    torch.cuda.empty_cache()
    test(model_, test_loader)

    torch.save(model_.state_dict(), "output/model.pth")
    log_thread.put(None)
    log_thread.join()

    plot_results("output/results.csv", "output/results.png")