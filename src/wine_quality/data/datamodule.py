"""Wine-quality dataset and dataloader factory.

This file is the data seam: `create_dataloaders(cfg, seed)` is all train.py
and eval.py know about. The knobs it needs (csv_path, batch size, split)
live on DataConfig in configs.py — both files are yours and are never
overwritten by `copier update`.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from wine_quality.configs import DataConfig
from wine_quality.utils.seed import seed_worker

WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)


class WineQualityDataset(Dataset):
    """UCI red wine quality: 11 physicochemical features -> quality score 3-8."""

    def __init__(self, csv_path: str) -> None:
        path = Path(csv_path)
        if not path.exists():  # download once, cache locally
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(WINE_URL, path)
        rows = np.loadtxt(path, delimiter=";", skiprows=1)
        x = torch.tensor(rows[:, :-1], dtype=torch.float32)
        self.data = (x - x.mean(0)) / x.std(0)  # standardize features
        self.targets = torch.tensor(rows[:, -1], dtype=torch.long) - 3  # 3-8 -> 0-5

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


@dataclass
class DataLoaders:
    """Train and val dataloaders."""

    train: DataLoader
    val: DataLoader


def create_dataloaders(cfg: DataConfig, seed: int = 42) -> DataLoaders:
    """Build train and val dataloaders from the data config — the data seam.

    Replace the dataset below with your own; keep this signature so train.py
    and eval.py never need editing. New knobs (paths, transforms, splits)
    belong on DataConfig.

    Args:
        cfg: The data config block.
        seed: Seed for the train/val split and worker init.

    Returns:
        DataLoaders with train and val loaders.
    """
    dataset = WineQualityDataset(cfg.csv_path)

    n_val = int(len(dataset) * cfg.val_split)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),  # pinning is CUDA-only; avoids the MPS warning
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),  # pinning is CUDA-only; avoids the MPS warning
        persistent_workers=cfg.num_workers > 0,
    )
    return DataLoaders(train=train_loader, val=val_loader)
