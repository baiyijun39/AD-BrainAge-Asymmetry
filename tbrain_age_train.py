import os
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

# Medical imaging (kept for compatibility)
import nibabel as nib 
import SimpleITK as sitk  

from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandAffine,
    RandBiasField,
    RandFlip,
    ToTensor,
)

# Project modules
from dataset import dataset


# Optional model summary
try:
    from torchsummary import summary  # type: ignore
except Exception:
    summary = None


# =========================
# Environment
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
torch.backends.cudnn.benchmark = True


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Config
# =========================
@dataclass(frozen=True)
class Config:
    data_path: str = "/raid/byj_file/T1_and_FLAIR_new/brain_age_data/data/byj/"
    out_dir: str = "/raid/byj_file/T1_and_FLAIR_new/out/"
    batch_size: int = 4
    num_workers: int = 4
    num_epochs: int = 150
    lr: float = 5e-5              
    momentum: float = 0.9
    seed: int = 42
    amp_enabled: bool = True


# =========================
# Dataset helper
# =========================
def dataseter(
    data: np.ndarray,
    labels: np.ndarray
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(len(labels)):
        dataset_list.append(
            (torch.tensor(data[i]), torch.tensor(labels[i]))
        )
    return dataset_list


def build_transforms() -> Compose:
    """
    与原代码一致：定义但不强制使用，避免改变原始行为
    """
    return Compose(
        [
            ToTensor(),
            RandFlip(prob=0.5, spatial_axis=0),
            RandAffine(prob=1.0, translate_range=(2, 2, 0)),
            RandBiasField(coeff_range=(0.1, 0.2), prob=0.5),
            NormalizeIntensity(channel_wise=True),
        ]
    )

def train_model(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    dataset_sizes: Dict[str, int],
    loss_fn: nn.Module,
    mae_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    device: torch.device,
    scaler: GradScaler,
    num_epochs: int,
    amp_enabled: bool,
):
    best_val_mae = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            is_train = phase == "train"
            model.train(is_train)

            running_loss = 0.0
            running_mae = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float().to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    with autocast(enabled=amp_enabled):
                        outputs = model(inputs).squeeze(-1)
                        loss = loss_fn(outputs, labels)
                        mae = mae_fn(outputs, labels)

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_mae += mae.item() * inputs.size(0)

            if is_train:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mae = running_mae / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_mae"].append(epoch_mae)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_mae:.4f}")

            if phase == "val" and epoch_mae < best_val_mae:
                best_val_mae = epoch_mae
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best val MAE: {best_val_mae:.6f}")

        print()

    print(f"Best val Acc: {best_val_mae:.6f}")
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ = build_transforms() 

    x_train, y_train, x_test, y_test = dataset(cfg.data_path)
    print("Downloaded train data")
    print("Downloaded validation data")

    train_ds = dataseter(x_train, y_train)
    val_ds = dataseter(x_test, y_test)

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        "val": torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
    }

    dataset_sizes = {
        "train": len(train_ds),
        "val": len(val_ds),
    }

    model = Small_SFCN().to(device)
    model = nn.DataParallel(model).to(device)

    if summary is not None:
        try:
            print(summary(model, (1, 121, 145, 121)))
        except Exception as e:
            print(f"Summary failed: {e}")

    loss_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
    )

    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
    )

    scaler = GradScaler(enabled=cfg.amp_enabled)

    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        loss_fn=loss_fn,
        mae_fn=mae_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        num_epochs=cfg.num_epochs,
        amp_enabled=cfg.amp_enabled,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    model_path = os.path.join(cfg.out_dir, f"densenetcam{cfg.lr}.pth")
    csv_path = os.path.join(cfg.out_dir, f"densenetcam{cfg.lr}.csv")

    torch.save(model, model_path)

    results = pd.DataFrame(history)
    results.to_csv(csv_path, index=False)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
