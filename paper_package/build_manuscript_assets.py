from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "dann_benchmark" / "runs"
PAPER_ROOT = PROJECT_ROOT / "paper_package"
MANUSCRIPT_ROOT = PAPER_ROOT / "manuscript"
FIG_ROOT = MANUSCRIPT_ROOT / "figures"
DERIVED_ROOT = PAPER_ROOT / "derived"
STATS_ROOT = PROJECT_ROOT / "stats_outputs"


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


MODEL_LABELS = {
    "dann_lrf": "DANN_LRF",
    "dann_random": "DANN_RANDOM",
    "dann_grf": "DANN_GRF",
    "naive_branch": "NAIVE_BRANCH",
    "mlp_param": "MLP_PARAM",
    "vann_same": "VANN_SAME",
}

MODEL_COLORS = {
    "dann_lrf": "#2C7FB8",
    "dann_random": "#74A9CF",
    "dann_grf": "#A6BDDB",
    "naive_branch": "#FDAE61",
    "mlp_param": "#D7191C",
    "vann_same": "#7F7F7F",
}

FULL_DATA_ORDER = ["mlp_param", "dann_random", "dann_grf", "naive_branch", "dann_lrf", "vann_same"]
LOW_DATA_ORDER = ["mlp_param", "naive_branch", "dann_lrf", "vann_same"]
TIMING_ORDER = ["mlp_param", "naive_branch", "dann_lrf", "vann_same"]


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    run_dir: Path
    label: str
    dataset: str
    subset_fraction: float
    family: str
    is_low_data: bool


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec("fashion_full", RUNS_ROOT / "runs_fashion_full" / "fashionmnist", "FashionMNIST full", "fashionmnist", 1.0, "full", False),
    ConditionSpec("kmnist_full", RUNS_ROOT / "runs_kmnist_full" / "kmnist", "KMNIST full", "kmnist", 1.0, "full", False),
    ConditionSpec("cifar_full", RUNS_ROOT / "runs_cifar_full" / "cifar10", "CIFAR-10 full", "cifar10", 1.0, "full", False),
    ConditionSpec("fashion_low02", RUNS_ROOT / "runs_fashion_low02" / "fashionmnist", "FashionMNIST 0.2", "fashionmnist", 0.2, "low", True),
    ConditionSpec("fashion_low01", RUNS_ROOT / "runs_fashion_low01" / "fashionmnist", "FashionMNIST 0.1", "fashionmnist", 0.1, "low", True),
    ConditionSpec("cifar_low02", RUNS_ROOT / "runs_cifar_low02" / "cifar10", "CIFAR-10 0.2", "cifar10", 0.2, "low", True),
)


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    STATS_ROOT.mkdir(parents=True, exist_ok=True)


def read_accuracy_records(spec: ConditionSpec) -> pd.DataFrame:
    history_dir = spec.run_dir / "histories"
    summary_path = spec.run_dir / "summary_by_seed.csv"
    config_path = spec.run_dir / "run_config.json"

    summary_df = pd.read_csv(summary_path)
    config = pd.read_json(config_path, typ="series")
    device = config["device"]

    records: list[dict[str, object]] = []
    for history_path in sorted(history_dir.glob("*.csv")):
        hist = pd.read_csv(history_path)
        model_name = str(hist["model_name"].iloc[0])
        seed = int(hist["seed"].iloc[0])
        best_val_idx = hist["val_loss"].idxmin()
        best_val_row = hist.loc[best_val_idx]

        summary_row = summary_df.loc[
            (summary_df["model_name"] == model_name) & (summary_df["seed"] == seed)
        ].iloc[0]

        records.append(
            {
                "condition_key": spec.key,
                "condition_label": spec.label,
                "dataset": spec.dataset,
                "subset_fraction": spec.subset_fraction,
                "device": device,
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "seed": seed,
                "trainable_params": int(summary_row["trainable_params"]),
                "best_val_epoch": int(best_val_row["epoch"]),
                "best_val_loss": float(best_val_row["val_loss"]),
                "test_acc_at_best_val": float(best_val_row["test_acc"]),
                "test_loss_at_best_val": float(best_val_row["test_loss"]),
                "best_test_acc": float(hist["test_acc"].max()),
                "min_test_loss": float(hist["test_loss"].min()),
                "final_test_acc": float(hist["test_acc"].iloc[-1]),
                "final_test_loss": float(hist["test_loss"].iloc[-1]),
            }
        )

    return pd.DataFrame.from_records(records)


def read_timing_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    timing_rows: list[pd.DataFrame] = []
    summary_rows: list[pd.DataFrame] = []
    timing_dirs = [
        RUNS_ROOT / "timing_fashion_full" / "fashionmnist",
        RUNS_ROOT / "timing_cifar_full" / "cifar10",
    ]
    for run_dir in timing_dirs:
        seed_path = run_dir / "timing_summary_by_seed.csv"
        model_path = run_dir / "timing_summary_by_model.csv"
        if seed_path.exists():
            seed_df = pd.read_csv(seed_path)
            seed_df["timing_condition"] = run_dir.parent.name
            timing_rows.append(seed_df)
        if model_path.exists():
            model_df = pd.read_csv(model_path)
            model_df["timing_condition"] = run_dir.parent.name
            summary_rows.append(model_df)

    return pd.concat(timing_rows, ignore_index=True), pd.concat(summary_rows, ignore_index=True)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    pvals = list(p_values)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * pvals[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_mean_ci(a: np.ndarray, b: np.ndarray, seed: int) -> tuple[float, float]:
    diffs = a - b
    rng = np.random.default_rng(seed)
    samples = diffs[rng.integers(0, len(diffs), size=(10000, len(diffs)))].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def build_paired_tests(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in CONDITIONS:
        cond_df = per_seed.loc[per_seed["condition_key"] == spec.key]
        comparisons = [("dann_lrf", "mlp_param"), ("dann_lrf", "naive_branch"), ("dann_lrf", "dann_random")]
        local_rows: list[dict[str, object]] = []
        for a_name, b_name in comparisons:
            a_df = cond_df.loc[cond_df["model_name"] == a_name].sort_values("seed")
            b_df = cond_df.loc[cond_df["model_name"] == b_name].sort_values("seed")
            if a_df.empty or b_df.empty:
                continue
            merged = a_df.merge(b_df, on="seed", suffixes=("_a", "_b"))
            a_vals = merged["test_acc_at_best_val_a"].to_numpy(dtype=float)
            b_vals = merged["test_acc_at_best_val_b"].to_numpy(dtype=float)
            if len(a_vals) == 0:
                continue
            stat = wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="two-sided", method="auto")
            ci_low, ci_high = bootstrap_mean_ci(a_vals, b_vals, seed=20260501 + len(rows))
            local_rows.append(
                {
                    "condition_key": spec.key,
                    "condition_label": spec.label,
                    "A": a_name,
                    "B": b_name,
                    "A_label": MODEL_LABELS[a_name],
                    "B_label": MODEL_LABELS[b_name],
                    "n": int(len(a_vals)),
                    "meanA": float(a_vals.mean()),
                    "meanB": float(b_vals.mean()),
                    "mean_diff": float((a_vals - b_vals).mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wilcoxon_W": float(stat.statistic),
                    "p_two_sided": float(stat.pvalue),
                }
            )
        if local_rows:
            adjusted = holm_adjust([row["p_two_sided"] for row in local_rows])
            for row, p_holm in zip(local_rows, adjusted):
                row["p_holm_per_condition"] = float(p_holm)
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def summarize_models(per_seed: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_seed.groupby(
            ["condition_key", "condition_label", "dataset", "subset_fraction", "device", "model_name", "model_label"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "count"),
            params_mean=("trainable_params", "mean"),
            test_acc_at_best_val_mean=("test_acc_at_best_val", "mean"),
            test_acc_at_best_val_std=("test_acc_at_best_val", "std"),
            best_test_acc_mean=("best_test_acc", "mean"),
            best_test_acc_std=("best_test_acc", "std"),
            final_test_acc_mean=("final_test_acc", "mean"),
            final_test_acc_std=("final_test_acc", "std"),
        )
        .sort_values(["condition_key", "test_acc_at_best_val_mean"], ascending=[True, False])
    )
    return grouped


def sensitivity_audit(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in CONDITIONS:
        cond = summary_df.loc[summary_df["condition_key"] == spec.key].copy()
        if cond.empty:
            continue
        primary_rank = cond.sort_values("test_acc_at_best_val_mean", ascending=False)["model_name"].tolist()
        historical_rank = cond.sort_values("best_test_acc_mean", ascending=False)["model_name"].tolist()
        rows.append(
            {
                "condition_key": spec.key,
                "condition_label": spec.label,
                "validation_selected_leader": MODEL_LABELS[primary_rank[0]],
                "historical_best_test_leader": MODEL_LABELS[historical_rank[0]],
                "leader_preserved": primary_rank[0] == historical_rank[0],
                "validation_selected_rank": " > ".join(MODEL_LABELS[name] for name in primary_rank),
                "historical_best_test_rank": " > ".join(MODEL_LABELS[name] for name in historical_rank),
            }
        )
    return pd.DataFrame.from_records(rows)


def save_plot(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, facecolor: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#333333",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=11)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="#333333"))


def plot_study_overview() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_axis_off()
    add_box(ax, (0.03, 0.58), 0.2, 0.25, "Scientific question\nWhen do dendritic\narchitectures help?", "#E8F1FA")
    add_box(ax, (0.28, 0.58), 0.2, 0.25, "Controlled models\nDANN_LRF, DANN_RANDOM,\nDANN_GRF, NAIVE_BRANCH,\nMLP_PARAM, VANN_SAME", "#FCEFD8")
    add_box(ax, (0.53, 0.58), 0.2, 0.25, "Benchmark conditions\nFashionMNIST, KMNIST,\nCIFAR-10,\nreduced-data diagnostics", "#EAF6E3")
    add_box(ax, (0.78, 0.58), 0.18, 0.25, "Reported evidence\nvalidation-selected\naccuracy, paired tests,\nand CPU timing", "#F6E8F4")
    add_arrow(ax, (0.23, 0.705), (0.28, 0.705))
    add_arrow(ax, (0.48, 0.705), (0.53, 0.705))
    add_arrow(ax, (0.73, 0.705), (0.78, 0.705))

    add_box(ax, (0.14, 0.18), 0.28, 0.2, "Fairness controls\nparameter matching\nand naive branching", "#FFF7E8")
    add_box(ax, (0.46, 0.18), 0.22, 0.2, "Primary contrasts\nDANN_LRF vs MLP_PARAM\nDANN_LRF vs NAIVE_BRANCH", "#EEF4FB")
    add_box(ax, (0.72, 0.18), 0.2, 0.2, "Interpretation scope\nparameter efficiency,\ndataset dependence,\nnot state of the art", "#F3F3F3")
    add_arrow(ax, (0.38, 0.38), (0.53, 0.58))
    add_arrow(ax, (0.57, 0.38), (0.63, 0.58))
    add_arrow(ax, (0.81, 0.38), (0.87, 0.58))
    ax.set_title("Study Overview and Claim Logic", pad=16)
    save_plot(fig, "fig_study_overview")


def plot_dann_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_axis_off()
    add_box(ax, (0.03, 0.25), 0.16, 0.46, "Flattened input image\nx in R^D", "#F4F4F4")
    add_box(ax, (0.25, 0.25), 0.2, 0.46, "Fixed feature routing\nLRF, RANDOM, or GRF\nK = 16 features per\ndendrite", "#E8F1FA")
    add_box(ax, (0.51, 0.25), 0.16, 0.46, "Dendrite stage\nlinear sum +\nLeakyReLU", "#DCEAF7")
    add_box(ax, (0.73, 0.25), 0.14, 0.46, "Soma stage\n4 branches per soma\nthen LeakyReLU", "#FCEFD8")
    add_box(ax, (0.91, 0.25), 0.06, 0.46, "10-way\nclassifier", "#EAF6E3")
    add_arrow(ax, (0.19, 0.48), (0.25, 0.48))
    add_arrow(ax, (0.45, 0.48), (0.51, 0.48))
    add_arrow(ax, (0.67, 0.48), (0.73, 0.48))
    add_arrow(ax, (0.87, 0.48), (0.91, 0.48))
    add_box(
        ax,
        (0.11, 0.02),
        0.34,
        0.13,
        "DANN variants differ only in how the fixed\nrouting tensor is sampled.",
        "#F5F5F5",
    )
    add_box(
        ax,
        (0.53, 0.02),
        0.36,
        0.13,
        "NAIVE_BRANCH keeps the routing but removes\ndendrite-level nonlinearity.",
        "#F5F5F5",
    )
    ax.set_title("DANN Layer and Baseline Logic", pad=16)
    save_plot(fig, "fig_dann_architecture")


def render_bar_panels(summary_df: pd.DataFrame, specs: list[ConditionSpec], stem: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(specs), figsize=(6.2 * len(specs), 5.2), sharey=False)
    if len(specs) == 1:
        axes = [axes]
    for ax, spec in zip(axes, specs):
        cond = summary_df.loc[summary_df["condition_key"] == spec.key].copy()
        order = [name for name in (LOW_DATA_ORDER if spec.is_low_data else FULL_DATA_ORDER) if name in cond["model_name"].values]
        cond["model_name"] = pd.Categorical(cond["model_name"], categories=order, ordered=True)
        cond = cond.sort_values("model_name")
        xs = np.arange(len(cond))
        ax.bar(
            xs,
            cond["test_acc_at_best_val_mean"],
            yerr=cond["test_acc_at_best_val_std"],
            capsize=4,
            color=[MODEL_COLORS[name] for name in cond["model_name"]],
            edgecolor="#333333",
            linewidth=1.0,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_LABELS[name] for name in cond["model_name"]], rotation=30, ha="right")
        ax.set_ylabel("Validation-selected test accuracy")
        ax.set_title(spec.label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_plot(fig, stem)


def plot_paired_effects(paired_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    plot_df = paired_df.copy()
    plot_df["comparison"] = plot_df["A_label"] + " - " + plot_df["B_label"]
    plot_df["row_label"] = plot_df["condition_label"] + "\n" + plot_df["comparison"]
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    colors = []
    for b_name in plot_df["B"]:
        colors.append({"mlp_param": "#2C7FB8", "naive_branch": "#FDAE61", "dann_random": "#8DA0CB"}[b_name])
    y = np.arange(len(plot_df))
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.errorbar(
        plot_df["mean_diff"],
        y,
        xerr=[plot_df["mean_diff"] - plot_df["ci95_low"], plot_df["ci95_high"] - plot_df["mean_diff"]],
        fmt="none",
        ecolor="#777777",
        elinewidth=2,
        capsize=4,
        zorder=1,
    )
    ax.scatter(plot_df["mean_diff"], y, s=210, c=colors, edgecolors="#222222", linewidths=1.4, zorder=2)
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        if row["p_holm_per_condition"] < 0.05:
            ax.text(row["mean_diff"] + 0.004, yi, "*", va="center", fontsize=16, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["row_label"])
    ax.set_xlabel("Mean paired accuracy difference (DANN_LRF - baseline)")
    ax.set_title("Seed-paired Effects with Bootstrap 95% Confidence Intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, "fig_paired_diffs")


def plot_timing_tradeoff(timing_model_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    panels = [("timing_fashion_full", "FashionMNIST timing"), ("timing_cifar_full", "CIFAR-10 timing")]
    for ax, (cond_key, title) in zip(axes, panels):
        cond = timing_model_df.loc[timing_model_df["timing_condition"] == cond_key].copy()
        cond = cond.sort_values("model_name", key=lambda s: [TIMING_ORDER.index(x) for x in s])
        for _, row in cond.iterrows():
            dx = 0.05
            dy = 0.3
            if row["model_name"] == "naive_branch":
                dx = 0.08
                dy = 0.55
            elif row["model_name"] == "dann_lrf":
                dx = 0.08
                dy = 0.05
            elif row["model_name"] == "vann_same":
                dx = 0.06
                dy = 0.45
            if cond_key == "timing_fashion_full" and row["model_name"] == "naive_branch":
                dx = -0.10
                dy = 0.18
            elif cond_key == "timing_fashion_full" and row["model_name"] == "dann_lrf":
                dx = 0.14
                dy = 0.08
            ax.scatter(
                row["train_seconds_per_epoch_mean"],
                row["inference_ms_per_1000_samples_mean"],
                s=350 + 900 * float(row["acc_mean"]),
                color=MODEL_COLORS[row["model_name"]],
                edgecolors="#222222",
                linewidths=1.5,
                alpha=0.9,
            )
            ax.text(
                row["train_seconds_per_epoch_mean"] + dx,
                row["inference_ms_per_1000_samples_mean"] + dy,
                MODEL_LABELS[row["model_name"]],
                fontsize=10,
            )
        ax.set_title(title)
        ax.set_xlabel("Train seconds per epoch (CPU)")
        ax.set_ylabel("Inference ms per 1000 samples (CPU)")
        ax.grid(alpha=0.25)
    fig.suptitle("CPU Timing Trade-offs", y=1.02)
    fig.tight_layout()
    save_plot(fig, "fig_timing")


def main() -> None:
    ensure_dirs()

    per_seed = pd.concat([read_accuracy_records(spec) for spec in CONDITIONS], ignore_index=True)
    summary_df = summarize_models(per_seed)
    sensitivity_df = sensitivity_audit(summary_df)
    paired_df = build_paired_tests(per_seed)
    timing_seed_df, timing_model_df = read_timing_records()

    per_seed.to_csv(DERIVED_ROOT / "validation_selected_by_seed.csv", index=False)
    summary_df.to_csv(DERIVED_ROOT / "validation_selected_summary_by_model.csv", index=False)
    sensitivity_df.to_csv(DERIVED_ROOT / "validation_selected_sensitivity_audit.csv", index=False)
    paired_df.to_csv(DERIVED_ROOT / "paired_tests_validation_selected.csv", index=False)
    paired_df.to_csv(STATS_ROOT / "paired_tests_validation_selected.csv", index=False)
    timing_seed_df.to_csv(DERIVED_ROOT / "timing_summary_by_seed.csv", index=False)
    timing_model_df.to_csv(DERIVED_ROOT / "timing_summary_by_model.csv", index=False)

    plot_study_overview()
    plot_dann_architecture()
    render_bar_panels(summary_df, list(CONDITIONS[:3]), "fig_full_data_bars", "Full-data Benchmark Results")
    render_bar_panels(summary_df, list(CONDITIONS[3:]), "fig_low_data_bars", "Reduced-data Diagnostic Results")
    plot_paired_effects(paired_df)
    plot_timing_tradeoff(timing_model_df)


if __name__ == "__main__":
    main()
