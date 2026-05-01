from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    test_loss: float
    test_acc: float


@dataclass
class RunSummary:
    dataset: str
    model_name: str
    seed: int
    trainable_params: int
    best_val_epoch: int
    best_val_loss: float
    test_acc_at_best_val: float
    test_loss_at_best_val: float
    best_test_acc: float
    min_test_loss: float
    final_test_acc: float
    final_test_loss: float


@dataclass
class EfficiencySummary:
    dataset: str
    model_name: str
    seed: int
    trainable_params: int
    best_val_epoch: int
    accuracy_efficiency: float
    loss_efficiency: float


DATASET_EPOCHS = {
    "mnist": 15,
    "fashionmnist": 20,
    "kmnist": 20,
    "emnist": 50,
    "cifar10": 50,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += x.size(0)
    return total_loss / total_examples, total_correct / total_examples


def train_one_run(
    model: nn.Module,
    loaders: Dict,
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
) -> Tuple[List[EpochMetrics], RunSummary]:
    set_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    model.to(device)

    history: List[EpochMetrics] = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_examples += x.size(0)

        train_loss = total_loss / total_examples
        train_acc = total_correct / total_examples
        val_loss, val_acc = evaluate(model, loaders["val"], device, criterion)
        test_loss, test_acc = evaluate(model, loaders["test"], device, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                test_loss=test_loss,
                test_acc=test_acc,
            )
        )

    best_val_metrics = min(history, key=lambda h: h.val_loss)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = RunSummary(
        dataset="",
        model_name="",
        seed=seed,
        trainable_params=trainable_params,
        best_val_epoch=best_val_metrics.epoch,
        best_val_loss=best_val_metrics.val_loss,
        test_acc_at_best_val=best_val_metrics.test_acc,
        test_loss_at_best_val=best_val_metrics.test_loss,
        best_test_acc=max(h.test_acc for h in history),
        min_test_loss=min(h.test_loss for h in history),
        final_test_acc=history[-1].test_acc,
        final_test_loss=history[-1].test_loss,
    )
    return history, summary


def compute_efficiency_summaries(
    summaries: List[RunSummary],
    accuracy_metric: str = "test_acc_at_best_val",
    loss_metric: str = "test_loss_at_best_val",
) -> List[EfficiencySummary]:
    if not summaries:
        return []
    min_k = min(max(1, s.trainable_params * s.best_val_epoch) for s in summaries)
    min_log_k = math.log10(min_k)
    out: List[EfficiencySummary] = []
    for s in summaries:
        k_i = max(1, s.trainable_params * s.best_val_epoch)
        f = math.log10(k_i) / min_log_k if min_log_k > 0 else 1.0
        acc_eff = float(getattr(s, accuracy_metric)) / f
        loss_eff = float(getattr(s, loss_metric)) * f
        out.append(
            EfficiencySummary(
                dataset=s.dataset,
                model_name=s.model_name,
                seed=s.seed,
                trainable_params=s.trainable_params,
                best_val_epoch=s.best_val_epoch,
                accuracy_efficiency=acc_eff,
                loss_efficiency=loss_eff,
            )
        )
    return out


def write_epoch_history_csv(path: Path, dataset: str, model_name: str, seed: int, history: List[EpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model_name", "seed", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"])
        for h in history:
            writer.writerow([
                dataset,
                model_name,
                seed,
                h.epoch,
                h.train_loss,
                h.train_acc,
                h.val_loss,
                h.val_acc,
                h.test_loss,
                h.test_acc,
            ])


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
