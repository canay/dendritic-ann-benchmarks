from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class ImageSpec:
    channels: int
    height: int
    width: int


DATASET_SPECS = {
    "fashionmnist": ImageSpec(channels=1, height=28, width=28),
    "kmnist": ImageSpec(channels=1, height=28, width=28),
    "cifar10": ImageSpec(channels=3, height=32, width=32),
}


def _feature_index(c: int, h: int, w: int, spec: ImageSpec) -> int:
    return c * spec.height * spec.width + h * spec.width + w


def _random_indices_from_patch(
    top: int,
    left: int,
    patch_h: int,
    patch_w: int,
    sample_size: int,
    spec: ImageSpec,
    g: torch.Generator,
) -> List[int]:
    patch_indices: List[int] = []
    for c in range(spec.channels):
        for h in range(top, min(spec.height, top + patch_h)):
            for w in range(left, min(spec.width, left + patch_w)):
                patch_indices.append(_feature_index(c, h, w, spec))
    if not patch_indices:
        raise ValueError("empty patch_indices")
    if len(patch_indices) >= sample_size:
        perm = torch.randperm(len(patch_indices), generator=g)[:sample_size]
        return [patch_indices[i] for i in perm.tolist()]
    # with replacement when the patch area is smaller than sample_size
    choice = torch.randint(low=0, high=len(patch_indices), size=(sample_size,), generator=g)
    return [patch_indices[i] for i in choice.tolist()]


def build_dendrite_indices(
    dataset_name: str,
    soma_units: int,
    branches_per_soma: int,
    sample_size: int,
    mode: str,
    seed: int,
) -> torch.Tensor:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    spec = DATASET_SPECS[dataset_name]
    num_dendrites = soma_units * branches_per_soma
    g = torch.Generator().manual_seed(int(seed))

    if mode == "random":
        input_dim = spec.channels * spec.height * spec.width
        return torch.randint(0, input_dim, (num_dendrites, sample_size), generator=g)

    patch_h, patch_w = 4, 4
    indices: List[List[int]] = []

    if mode == "lrf":
        for _ in range(num_dendrites):
            top = int(torch.randint(0, spec.height - patch_h + 1, (1,), generator=g).item())
            left = int(torch.randint(0, spec.width - patch_w + 1, (1,), generator=g).item())
            indices.append(_random_indices_from_patch(top, left, patch_h, patch_w, sample_size, spec, g))

    elif mode == "grf":
        for _soma in range(soma_units):
            top = int(torch.randint(0, spec.height - patch_h + 1, (1,), generator=g).item())
            left = int(torch.randint(0, spec.width - patch_w + 1, (1,), generator=g).item())
            for _branch in range(branches_per_soma):
                indices.append(_random_indices_from_patch(top, left, patch_h, patch_w, sample_size, spec, g))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return torch.tensor(indices, dtype=torch.long)
