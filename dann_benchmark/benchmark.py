from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import torch

from src.data import make_dataloaders
from src.models import (
    DendriticANN,
    FlatMLP,
    NaiveBranchedLinear,
    VanillaANN,
    count_parameters,
    estimate_param_matched_width,
)
from src.sampling import build_dendrite_indices
from src.train_eval import (
    DATASET_EPOCHS,
    RunSummary,
    compute_efficiency_summaries,
    set_seed,
    train_one_run,
    write_epoch_history_csv,
    write_json,
)


DEFAULT_MODELS = ["dann_random", "dann_lrf", "dann_grf", "naive_branch", "vann_same", "mlp_param"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight benchmark for dendritic ANNs vs simple controls.")
    p.add_argument("--dataset", type=str, required=True, choices=["fashionmnist", "kmnist", "cifar10"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./runs")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--subset-fraction", type=float, default=1.0)
    p.add_argument("--test-subset-fraction", type=float, default=None)
    p.add_argument("--soma-units", type=int, default=128)
    p.add_argument("--branches-per-soma", type=int, default=4)
    p.add_argument("--sample-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--report-metric",
        type=str,
        default="test_acc_at_best_val",
        choices=["test_acc_at_best_val", "final_test_acc", "best_test_acc"],
    )
    return p.parse_args()


def select_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def build_model(
    model_name: str,
    dataset: str,
    input_dim: int,
    num_classes: int,
    soma_units: int,
    branches_per_soma: int,
    sample_size: int,
    seed: int,
):
    num_dendrites = soma_units * branches_per_soma
    if model_name == "dann_random":
        idx = build_dendrite_indices(dataset, soma_units, branches_per_soma, sample_size, mode="random", seed=seed)
        model = DendriticANN(input_dim, num_classes, soma_units, branches_per_soma, sample_size, idx)
    elif model_name == "dann_lrf":
        idx = build_dendrite_indices(dataset, soma_units, branches_per_soma, sample_size, mode="lrf", seed=seed)
        model = DendriticANN(input_dim, num_classes, soma_units, branches_per_soma, sample_size, idx)
    elif model_name == "dann_grf":
        idx = build_dendrite_indices(dataset, soma_units, branches_per_soma, sample_size, mode="grf", seed=seed)
        model = DendriticANN(input_dim, num_classes, soma_units, branches_per_soma, sample_size, idx)
    elif model_name == "naive_branch":
        idx = build_dendrite_indices(dataset, soma_units, branches_per_soma, sample_size, mode="lrf", seed=seed)
        model = NaiveBranchedLinear(input_dim, num_classes, soma_units, branches_per_soma, sample_size, idx)
    elif model_name == "vann_same":
        model = VanillaANN(input_dim, num_classes, num_dendrites, soma_units)
    elif model_name == "mlp_param":
        # Match the parameter count of the LRF dANN, which is the main comparison target.
        idx = build_dendrite_indices(dataset, soma_units, branches_per_soma, sample_size, mode="lrf", seed=seed)
        proxy = DendriticANN(input_dim, num_classes, soma_units, branches_per_soma, sample_size, idx)
        target_params = count_parameters(proxy)
        width = estimate_param_matched_width(target_params, input_dim, num_classes)
        model = FlatMLP(input_dim, num_classes, width)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model


def paired_loss_metric(accuracy_metric: str) -> str:
    mapping = {
        "test_acc_at_best_val": "test_loss_at_best_val",
        "final_test_acc": "final_test_loss",
        "best_test_acc": "min_test_loss",
    }
    return mapping[accuracy_metric]


def summarize_by_model(rows: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model_name"]].append(row)

    out: List[dict] = []
    for model_name, items in grouped.items():
        reported_metric = str(items[0]["reported_metric"])
        reported_loss_metric = str(items[0]["reported_loss_metric"])
        reported_accs = [float(x["reported_test_acc"]) for x in items]
        reported_losses = [float(x["reported_test_loss"]) for x in items]
        val_selected_accs = [float(x["test_acc_at_best_val"]) for x in items]
        best_accs = [float(x["best_test_acc"]) for x in items]
        final_accs = [float(x["final_test_acc"]) for x in items]
        aes = [float(x["accuracy_efficiency"]) for x in items]
        les = [float(x["loss_efficiency"]) for x in items]
        params = [int(x["trainable_params"]) for x in items]

        out.append(
            {
                "model_name": model_name,
                "n_seeds": len(items),
                "params_mean": mean(params),
                "reported_metric": reported_metric,
                "reported_loss_metric": reported_loss_metric,
                "reported_test_acc_mean": mean(reported_accs),
                "reported_test_acc_std": stdev(reported_accs) if len(reported_accs) > 1 else 0.0,
                "reported_test_loss_mean": mean(reported_losses),
                "reported_test_loss_std": stdev(reported_losses) if len(reported_losses) > 1 else 0.0,
                "test_acc_at_best_val_mean": mean(val_selected_accs),
                "test_acc_at_best_val_std": stdev(val_selected_accs) if len(val_selected_accs) > 1 else 0.0,
                "best_test_acc_mean": mean(best_accs),
                "best_test_acc_std": stdev(best_accs) if len(best_accs) > 1 else 0.0,
                "final_test_acc_mean": mean(final_accs),
                "final_test_acc_std": stdev(final_accs) if len(final_accs) > 1 else 0.0,
                "accuracy_efficiency_mean": mean(aes),
                "accuracy_efficiency_std": stdev(aes) if len(aes) > 1 else 0.0,
                "loss_efficiency_mean": mean(les),
                "loss_efficiency_std": stdev(les) if len(les) > 1 else 0.0,
            }
        )

    return sorted(out, key=lambda x: (-x["reported_test_acc_mean"], x["params_mean"]))


def make_decision_report(summary_rows: List[dict]) -> dict:
    by_name = {row["model_name"]: row for row in summary_rows}
    target_candidates = [name for name in ["dann_lrf", "dann_grf", "dann_random"] if name in by_name]
    verdict = {
        "decision": "inconclusive",
        "why": [],
    }
    if not target_candidates:
        verdict["why"].append("No dANN model was executed.")
        return verdict

    best_dann_name = max(target_candidates, key=lambda n: by_name[n]["reported_test_acc_mean"])
    best_dann = by_name[best_dann_name]

    checks = {
        "beats_naive_branch_acc": False,
        "beats_param_mlp_acc": False,
        "beats_vann_same_efficiency": False,
        "stable_enough": False,
    }
    if "naive_branch" in by_name:
        checks["beats_naive_branch_acc"] = best_dann["reported_test_acc_mean"] > by_name["naive_branch"]["reported_test_acc_mean"]
    if "mlp_param" in by_name:
        checks["beats_param_mlp_acc"] = best_dann["reported_test_acc_mean"] >= by_name["mlp_param"]["reported_test_acc_mean"]
    if "vann_same" in by_name:
        checks["beats_vann_same_efficiency"] = best_dann["accuracy_efficiency_mean"] > by_name["vann_same"]["accuracy_efficiency_mean"]
    checks["stable_enough"] = best_dann["reported_test_acc_std"] <= 0.01

    passed = sum(bool(v) for v in checks.values())
    if passed >= 3:
        verdict["decision"] = "continue"
        verdict["why"].append(f"{best_dann_name} passed {passed}/4 quick-kill checks.")
    else:
        verdict["decision"] = "stop_or_rethink"
        verdict["why"].append(f"{best_dann_name} passed only {passed}/4 quick-kill checks.")
    verdict["checks"] = checks
    verdict["best_dann"] = best_dann_name
    verdict["reported_metric"] = summary_rows[0]["reported_metric"] if summary_rows else None
    return verdict


def main() -> None:
    args = parse_args()
    epochs = args.epochs or DATASET_EPOCHS[args.dataset]
    device = select_device(args.device)
    report_loss_metric = paired_loss_metric(args.report_metric)

    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    all_run_summaries: List[RunSummary] = []
    print(f"[info] dataset={args.dataset} epochs={epochs} device={device} models={args.models} seeds={args.seeds}")

    for seed in args.seeds:
        set_seed(seed)
        loaders, info = make_dataloaders(
            dataset_name=args.dataset,
            root=args.data_root,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
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
            history, summary = train_one_run(model, loaders, device, epochs=epochs, lr=args.lr, seed=seed)
            summary.dataset = args.dataset
            summary.model_name = model_name
            all_run_summaries.append(summary)
            write_epoch_history_csv(out_root / "histories" / f"{model_name}_seed{seed}.csv", args.dataset, model_name, seed, history)

    eff_summaries = compute_efficiency_summaries(
        all_run_summaries,
        accuracy_metric=args.report_metric,
        loss_metric=report_loss_metric,
    )
    eff_map = {(e.dataset, e.model_name, e.seed): e for e in eff_summaries}

    for summary in all_run_summaries:
        eff = eff_map[(summary.dataset, summary.model_name, summary.seed)]
        all_rows.append(
            {
                "dataset": summary.dataset,
                "model_name": summary.model_name,
                "seed": summary.seed,
                "trainable_params": summary.trainable_params,
                "best_val_epoch": summary.best_val_epoch,
                "best_val_loss": summary.best_val_loss,
                "test_acc_at_best_val": summary.test_acc_at_best_val,
                "test_loss_at_best_val": summary.test_loss_at_best_val,
                "best_test_acc": summary.best_test_acc,
                "min_test_loss": summary.min_test_loss,
                "final_test_acc": summary.final_test_acc,
                "final_test_loss": summary.final_test_loss,
                "reported_metric": args.report_metric,
                "reported_loss_metric": report_loss_metric,
                "reported_test_acc": getattr(summary, args.report_metric),
                "reported_test_loss": getattr(summary, report_loss_metric),
                "accuracy_efficiency": eff.accuracy_efficiency,
                "loss_efficiency": eff.loss_efficiency,
            }
        )

    summary_csv = out_root / "summary_by_seed.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    model_summary = summarize_by_model(all_rows)
    with (out_root / "summary_by_model.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_summary[0].keys()))
        writer.writeheader()
        writer.writerows(model_summary)

    decision_report = make_decision_report(model_summary)
    write_json(out_root / "decision_report.json", decision_report)
    write_json(out_root / "run_config.json", vars(args))

    print("\n[summary]")
    for row in model_summary:
        print(
            f"{row['model_name']:>12} | acc={row['reported_test_acc_mean']:.4f}+/-{row['reported_test_acc_std']:.4f} "
            f"| les={row['loss_efficiency_mean']:.4f} | aes={row['accuracy_efficiency_mean']:.4f} "
            f"| params~{row['params_mean']:.0f} | metric={row['reported_metric']}"
        )
    print("\n[decision]", json.dumps(decision_report, indent=2))


if __name__ == "__main__":
    main()
