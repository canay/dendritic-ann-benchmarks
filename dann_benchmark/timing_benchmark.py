from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from benchmark import build_model, select_device
from src.data import make_dataloaders
from src.models import count_parameters
from src.train_eval import set_seed, train_one_run, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["fashionmnist", "kmnist", "cifar10"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./timing_runs")
    p.add_argument("--models", nargs="+", default=["dann_lrf", "naive_branch", "mlp_param"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--soma-units", type=int, default=128)
    p.add_argument("--branches-per-soma", type=int, default=4)
    p.add_argument("--sample-size", type=int, default=16)
    p.add_argument("--subset-fraction", type=float, default=1.0)
    p.add_argument("--test-subset-fraction", type=float, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


@torch.no_grad()
def measure_inference(model, loader, device, repeats=5):
    model.eval()
    times = []
    n_samples = 0

    for x, _ in loader:
        x = x.to(device)
        _ = model(x)
        break

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(repeats):
        start = time.perf_counter()
        total = 0

        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            total += x.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        n_samples = total

    avg_time = sum(times) / len(times)
    return {
        "inference_total_seconds": avg_time,
        "inference_samples": n_samples,
        "inference_ms_per_1000_samples": (avg_time / n_samples) * 1000 * 1000,
    }


def build_timing_config(args, device) -> dict:
    payload = {
        **vars(args),
        "selected_device": str(device),
        "accuracy_metric": "test_acc_at_best_val",
        "purpose": "Relative timing benchmark for train and inference cost under a fixed protocol.",
        "runtime_notes": {
            "timing_interpretation": "Use as relative timing evidence under the recorded protocol, not as a universal hardware claim.",
            "hardware_environment_note": "Add exact CPU or GPU host details separately if needed; the script records only the selected runtime device.",
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads(),
        },
    }
    if device.type == "cuda":
        payload["environment"]["cuda_device_name"] = torch.cuda.get_device_name(device)
    return payload


def main():
    args = parse_args()
    device = select_device(args.device)

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "timing_run_config.json", build_timing_config(args, device))

    rows = []

    print(f"[info] dataset={args.dataset} device={device} epochs={args.epochs} models={args.models} seeds={args.seeds}")

    for seed in args.seeds:
        set_seed(seed)
        loaders, info = make_dataloaders(
            dataset_name=args.dataset,
            root=args.data_root,
            batch_size=args.batch_size,
            val_fraction=0.1,
            subset_fraction=args.subset_fraction,
            test_subset_fraction=args.test_subset_fraction,
            seed=seed,
            num_workers=args.num_workers,
        )

        for model_name in args.models:
            print(f"[run] seed={seed} model={model_name}")
            set_seed(seed)
            model = build_model(
                model_name=model_name,
                dataset=args.dataset,
                input_dim=info.input_dim,
                num_classes=info.num_classes,
                soma_units=args.soma_units,
                branches_per_soma=args.branches_per_soma,
                sample_size=args.sample_size,
                seed=seed,
            )

            params = count_parameters(model)

            start = time.perf_counter()
            history, summary = train_one_run(
                model,
                loaders,
                device,
                epochs=args.epochs,
                lr=args.lr,
                seed=seed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_total = time.perf_counter() - start

            infer = measure_inference(model.to(device), loaders["test"], device)

            rows.append(
                {
                    "dataset": args.dataset,
                    "model_name": model_name,
                    "seed": seed,
                    "epochs": args.epochs,
                    "trainable_params": params,
                    "train_total_seconds": train_total,
                    "train_seconds_per_epoch": train_total / args.epochs,
                    "test_acc_at_best_val": summary.test_acc_at_best_val,
                    "best_test_acc": summary.best_test_acc,
                    "final_test_acc": summary.final_test_acc,
                    "min_test_loss": summary.min_test_loss,
                    **infer,
                }
            )

    out_csv = out_dir / "timing_summary_by_seed.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    model_names = sorted(set(r["model_name"] for r in rows))
    summary_rows = []

    for model_name in model_names:
        items = [r for r in rows if r["model_name"] == model_name]

        def avg(key):
            return sum(float(x[key]) for x in items) / len(items)

        summary_rows.append(
            {
                "dataset": args.dataset,
                "model_name": model_name,
                "n_seeds": len(items),
                "params_mean": avg("trainable_params"),
                "acc_metric": "test_acc_at_best_val",
                "acc_mean": avg("test_acc_at_best_val"),
                "train_seconds_per_epoch_mean": avg("train_seconds_per_epoch"),
                "train_total_seconds_mean": avg("train_total_seconds"),
                "inference_ms_per_1000_samples_mean": avg("inference_ms_per_1000_samples"),
            }
        )

    out_summary = out_dir / "timing_summary_by_model.csv"
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n[done]")
    print(out_summary)


if __name__ == "__main__":
    main()
