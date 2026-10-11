"""Cached-tensor data pipeline that reproduces ``src.data.make_dataloaders`` exactly.

The r0 pipeline converts every image with ``transforms.ToTensor()`` on every
access. Here each torchvision dataset is converted ONCE with the very same
transform, flattened exactly like ``FlattenWrapper`` (``x.view(-1)``), and kept
as one contiguous float32 tensor. Subset selection, the fit/validation split and
the DataLoader configuration are the r0 calls with the same arguments, so the
index sets, the shuffling RNG draws and every batch tensor are identical. The
equivalence is not assumed: ``golden_check.py`` compares epoch histories of the
r0 path and this path byte for byte on CPU before any manuscript-facing run.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.data import DatasetInfo, _subset_dataset, _torchvision_datasets
from src.sampling import DATASET_SPECS


class CachedFlatDataset(Dataset):
    """(x, y) pairs from pre-converted tensors; ``x`` is the flattened ToTensor output."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, perm: Optional[torch.Tensor] = None) -> None:
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("cached tensors have unexpected shapes")
        self.x = x if perm is None else x[:, perm].contiguous()
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        # torchvision returns the label as a Python int; default_collate turns a
        # list of ints into an int64 tensor, which is what the r0 loaders produced.
        return self.x[idx], int(self.y[idx])


def _convert(base: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for i in range(len(base)):
        x, y = base[i]
        xs.append(x.view(-1))
        ys.append(int(y))
    return torch.stack(xs).contiguous(), torch.tensor(ys, dtype=torch.int64)


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest().upper()


_MEMORY_CACHE: Dict[str, Dict[str, object]] = {}


def load_cached(dataset_name: str, root: str) -> Dict[str, object]:
    """Return {'train_x','train_y','test_x','test_y','num_classes','hashes'}.

    The conversion is kept in process memory only (no disk cache): the shared
    experiment host has little free disk, and one conversion per worker process
    and dataset costs seconds.
    """
    name = dataset_name.lower()
    if name in _MEMORY_CACHE:
        return _MEMORY_CACHE[name]
    train_base, test_base, num_classes = _torchvision_datasets(name, root)
    tx, ty = _convert(train_base)
    ex, ey = _convert(test_base)
    blob = {"train_x": tx, "train_y": ty, "test_x": ex, "test_y": ey}
    hashes = {k: tensor_sha256(v) for k, v in blob.items()}
    out: Dict[str, object] = dict(blob)
    out["num_classes"] = int(num_classes)
    out["hashes"] = hashes
    _MEMORY_CACHE[name] = out
    return out


def make_cached_dataloaders(
    cached: Dict[str, object],
    dataset_name: str,
    batch_size: int,
    val_fraction: float,
    subset_fraction: float,
    data_seed: int,
    num_workers: int = 0,
    pixel_perm: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, DataLoader], DatasetInfo]:
    """Mirror of ``src.data.make_dataloaders`` (test set always full, as in the fixed-test r0 runs).

    ``pixel_perm`` (optional) applies one fixed permutation of the flattened input
    features to train, validation and test alike (shuffled-pixel control).
    """
    train_ds = CachedFlatDataset(cached["train_x"], cached["train_y"], pixel_perm)
    test_ds = CachedFlatDataset(cached["test_x"], cached["test_y"], pixel_perm)
    train_base = _subset_dataset(train_ds, subset_fraction, data_seed)

    n_train = len(train_base)
    n_val = max(1, int(n_train * val_fraction))
    n_fit = n_train - n_val
    if n_fit < 1:
        raise ValueError("Training split is empty. Reduce val_fraction.")
    g = torch.Generator().manual_seed(data_seed)
    fit_ds, val_ds = random_split(train_base, [n_fit, n_val], generator=g)

    train_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    spec = DATASET_SPECS[dataset_name.lower()]
    input_dim = spec.channels * spec.height * spec.width
    info = DatasetInfo(name=dataset_name.lower(), input_dim=input_dim, num_classes=int(cached["num_classes"]))
    return {"train": train_loader, "val": val_loader, "test": test_loader}, info
