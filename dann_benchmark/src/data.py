from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .sampling import DATASET_SPECS


@dataclass
class DatasetInfo:
    name: str
    input_dim: int
    num_classes: int


def _flatten_item(sample):
    x, y = sample
    return x.view(-1), y


class FlattenWrapper(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x.view(-1), y


def _torchvision_datasets(dataset_name: str, root: str):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "torchvision import failed. Install a compatible torch/torchvision pair. "
            "Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        ) from exc

    transform = transforms.ToTensor()
    dataset_name = dataset_name.lower()
    if dataset_name == "fashionmnist":
        train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "kmnist":
        train = datasets.KMNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.KMNIST(root=root, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "cifar10":
        train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return train, test, num_classes


def _subset_dataset(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    if fraction >= 0.9999:
        return dataset
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = len(dataset)
    k = max(1, int(n * fraction))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)[:k].tolist()
    return Subset(dataset, perm)


def make_dataloaders(
    dataset_name: str,
    root: str,
    batch_size: int,
    val_fraction: float,
    subset_fraction: float,
    seed: int,
    num_workers: int = 0,
    test_subset_fraction: float | None = None,
) -> Tuple[Dict[str, DataLoader], DatasetInfo]:
    train_base, test_base, num_classes = _torchvision_datasets(dataset_name, root)
    train_base = _subset_dataset(train_base, subset_fraction, seed)
    if test_subset_fraction is not None:
        # Keep this optional for legacy reproduction, but default to the full test set.
        test_base = _subset_dataset(test_base, test_subset_fraction, seed + 999)

    train_wrapped = FlattenWrapper(train_base)
    test_wrapped = FlattenWrapper(test_base)

    n_train = len(train_wrapped)
    n_val = max(1, int(n_train * val_fraction))
    n_fit = n_train - n_val
    if n_fit < 1:
        raise ValueError("Training split is empty. Reduce val_fraction.")

    g = torch.Generator().manual_seed(seed)
    fit_ds, val_ds = random_split(train_wrapped, [n_fit, n_val], generator=g)

    train_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    spec = DATASET_SPECS[dataset_name.lower()]
    input_dim = spec.channels * spec.height * spec.width
    info = DatasetInfo(name=dataset_name.lower(), input_dim=input_dim, num_classes=num_classes)
    return {"train": train_loader, "val": val_loader, "test": test_loader}, info
