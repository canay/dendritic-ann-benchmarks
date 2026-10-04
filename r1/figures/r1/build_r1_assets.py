"""Build the R1 manuscript tables (LaTeX row bodies) and Figures 3-4 of NEURO/manuscript-r1 from verified outputs.

Inputs are read only from the verified processed outputs of the three R1 runs and, for the descriptive fit
diagnostics of protocol F8(c), from the verified raw archive of the frozen run. Nothing is trained or
re-estimated; the F8(c) summaries are descriptive means over the frozen histories (the frozen analysis manifest
did not emit them).

Outputs
  figures/r1/out/table3_families.tex, table4_accuracy.tex, table5_contrasts.tex, table6_references.tex,
  table7_timing.tex, f8c_fit_diagnostics.csv, figure_data_*.csv, text_numbers.json, PROVENANCE.json
  figures/r1/out/fig_sensitivity.pdf, fig_controls.pdf (vector twins for the font gate)
  NEURO/manuscript-r1/fig_sensitivity.png, fig_controls.png (600 dpi TeX-facing carriers)

Fonts follow METHODOLOGY_OVERVIEW_FIGURE.md (Fig1 type roles): axis and label text IBM Plex Sans SemiBold, normal
text Barlow Regular; both are loaded by path and the loaded family is asserted.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("error", message=r".*[Gg]lyph.*missing.*")  # a missing glyph is a defect, not a fallback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

P = Path(__file__).resolve().parents[2]
FROZEN = P / "runs" / "2026-10-03_claude_mta_cuda_r1_frozen"
FROZEN_PO = FROZEN / "processed_outputs"
FROZEN_ARCHIVE = FROZEN / "raw_outputs" / "2026-10-03_claude_mta_cuda_r1_frozen__public_outputs.tar.gz"  # public copy
F10_PO = P / "runs" / "2026-10-04_claude_mta_cuda_r1_f10_gpu" / "processed_outputs"
ARM_PO = P / "runs" / "2026-10-04_claude_mta_cuda_r1_cnnflat" / "processed_outputs"
OUT = P / "figures" / "r1" / "out"
MS = P / "figures" / "r1" / "out"  # public copy: PNG carriers go next to the PDFs
FONT_DIR = Path(os.environ.get("R1_FONT_DIR", str(P / "fonts")))  # public copy
FONT_LABEL = FONT_DIR / "IBMPlexSans-SemiBold.ttf"  # IBM Plex Sans SemiBold (role 2)
FONT_TEXT = FONT_DIR / "Barlow-Regular.ttf"  # Barlow Regular (role 3)

PALETTE = {"navy": "#17324D", "slate": "#4F6B7A", "light": "#8FA7B5", "pale": "#EAF0F4", "ink": "#1F2933",
           "amber": "#9C6B30"}
DATASET_LABEL = {"fashionmnist": "FashionMNIST", "kmnist": "KMNIST", "cifar10": "CIFAR-10"}
COND_DATASET = {"fashion_full": "fashionmnist", "kmnist_full": "kmnist", "cifar_full": "cifar10",
                "fashion_low02": "fashionmnist", "fashion_low01": "fashionmnist", "cifar_low02": "cifar10"}
CONDITIONS = ["fashion_full", "kmnist_full", "cifar_full", "fashion_low02", "fashion_low01", "cifar_low02"]
COND_LABEL = {"fashion_full": "FashionMNIST full", "kmnist_full": "KMNIST full", "cifar_full": "CIFAR-10 full",
              "fashion_low02": "FashionMNIST 0.2", "fashion_low01": "FashionMNIST 0.1",
              "cifar_low02": "CIFAR-10 0.2"}
INPUTS: list[Path] = []
OUTPUTS: list[Path] = []
NUMBERS: dict[str, object] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> pd.DataFrame:
    INPUTS.append(path)
    return pd.read_csv(path)


def fmt(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        if float(s) == 0.0:
            return s[1:]
        return "$-$" + s[1:]
    return s


def pm(mean: float, sd: float) -> str:
    return f"{fmt(mean)} $\\pm$ {fmt(sd)}"


def ci(lo: float, hi: float) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def thousands(n: int) -> str:
    return f"{int(n):,}"


LABELS = {"supported": "supported", "not_supported": "not supported", "reverse": "reverse",
          "no_difference": "no difference"}


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    OUTPUTS.append(path)


# ----------------------------------------------------------------------------------------------- inputs
summary = read_csv(FROZEN_PO / "summary_by_cell.csv")
decision = read_csv(FROZEN_PO / "decision_inputs.csv")
descriptive = read_csv(FROZEN_PO / "descriptive.csv")
units = read_csv(FROZEN_PO / "unit_table.csv")
replication = read_csv(FROZEN_PO / "replication_r0_vs_r1.csv")
composite = read_csv(FROZEN_PO / "composite_decisions.csv")
f10 = read_csv(F10_PO / "f10_summary.csv")
f10_ratios = read_csv(F10_PO / "f10_ratios.csv")
arm = read_csv(ARM_PO / "arm_summary.csv")
arm_paired = read_csv(ARM_PO / "arm_paired.csv")
assert len(decision) == 39, len(decision)
assert len(units) == 1615, len(units)


def cell(family: str, condition: str, model: str) -> tuple[float, float, int]:
    rows = summary[(summary.family == family) & (summary.condition == condition) & (summary.model == model)]
    assert len(rows) == 1, (family, condition, model, len(rows))
    r = rows.iloc[0]
    return float(r.acc_mean) * 100.0, float(r.acc_sd) * 100.0, int(r.n_seeds)


# parameters: effective trainable parameters per (model, dataset) from the frozen unit table
params: dict[tuple[str, str], int] = {}
for (model, dataset), grp in units[units.family != "F3_sensitivity"].groupby(["model", "dataset"]):
    values = set(int(v) for v in grp.effective_params)
    assert len(values) == 1, (model, dataset, values)
    params[(model, dataset)] = values.pop()
for dataset_key, dataset_f10 in (("fashionmnist", "fashionmnist"), ("cifar10", "cifar10")):
    vals = set(int(v) for v in f10[(f10.model == "cnn_flat") & (f10.dataset == dataset_f10)].trainable_params_effective)
    assert len(vals) == 1, vals
    params[("cnn_flat", dataset_key)] = vals.pop()
params[("cnn_flat", "kmnist")] = params[("cnn_flat", "fashionmnist")]  # same 28x28x1 input shape (A12)
for model in ("dann_lrf", "naive_branch", "mlp_param", "vann_same", "lc_net", "sparse_mlp", "compact_cnn",
              "dann_random", "dann_grf"):
    assert params[(model, "fashionmnist")] == params[(model, "kmnist")], model

# ----------------------------------------------------------------------------------------------- Table 3
fam_counts = units.groupby("family").size().to_dict()
NUMBERS["units_per_family"] = {k: int(v) for k, v in fam_counts.items()}
f8a = units[(units.family == "F8_fairness") & (units.model != "naive_branch_matched_init")]
f8b = units[(units.family == "F8_fairness") & (units.model == "naive_branch_matched_init")]
NUMBERS["units_f8a"] = int(len(f8a))
NUMBERS["units_f8b"] = int(len(f8b))
f10_units = int(f10.seeds.sum())  # one unit per (device, dataset, model, seed)
assert f10_units == 102, f10_units
NUMBERS["units_f10"] = f10_units
NUMBERS["units_arm"] = 66
families = [
    ("F1", "Main grid", "RQ1, RQ2", "nonlinearity and dense contrasts; routing variants",
     "\\DANNLRF{}, \\DANNRANDOM{}, \\DANNGRF{}, \\NAIVEBRANCH{}, \\MLPPARAM{}, \\VANNSAME{}",
     "3 full; 3 reduced", "0--19", fam_counts["main_grid"]),
    ("F2", "Reference priors", "RQ2", "what another prior buys at the same budget",
     "compact CNN, LC network, random-sparse MLP, MLP-Matched", "3 full; CIFAR-10 0.2", "0--19",
     fam_counts["F2_baselines"]),
    ("F3", "Sensitivity", "RQ1, RQ2", "operating point $(K, B)$ and patch size (descriptive)",
     "\\DANNLRF{}; \\DANNRANDOM{} on the KMNIST grid", "FashionMNIST, KMNIST full", "0--4",
     fam_counts["F3_sensitivity"]),
    ("F4", "Randomness sources", "RQ1", "routing, initialization and data seeds varied one at a time (descriptive)",
     "\\DANNLRF{}, \\NAIVEBRANCH{}", "FashionMNIST full", "0--9 per source", fam_counts["F4_randomness"]),
    ("F5", "Shuffled pixels", "RQ2", "locality of the routing prior",
     "\\DANNLRF{}, \\DANNRANDOM{}, \\NAIVEBRANCH{}, \\MLPPARAM{}", "FashionMNIST, CIFAR-10 full", "0--19",
     fam_counts["F5_shuffled"]),
    ("F6", "Channel-aware routing", "RQ2", "single-channel versus cross-channel patches",
     "\\DANNLRF{} variant", "CIFAR-10 full, 0.2", "0--19", fam_counts["F6_channel"]),
    ("F7", "Dendrite slope", "RQ1", "dose-response of the dendrite activation (descriptive)",
     "\\DANNLRF{}, $\\alpha_d \\in \\{0, \\ldots, 1\\}$", "3 full", "0--9", fam_counts["F7_slope"]),
    ("F8", "Optimization fairness", "RQ1", "learning rate (descriptive); matched initialization",
     "\\DANNLRF{}, \\NAIVEBRANCH{}", "3 full", "0--4; 0--19", fam_counts["F8_fairness"]),
    ("F9", "Convolutional stem", "RQ1, RQ2", "transfer of the attribution to a convolutional front end",
     "shared stem with \\DANNLRF{}, \\NAIVEBRANCH{} and MLP heads", "FashionMNIST, CIFAR-10 full", "0--19",
     fam_counts["F9_convstem"]),
    ("F10", "Timing", "RQ3", "training and inference cost on CPU and GPU (descriptive)",
     "nine models", "FashionMNIST, CIFAR-10 full", "0--2, 5 repeats", f10_units),
    ("A12", "Spatial-head CNN$^{\\dagger}$", "RQ2", "exploratory reference with a fitted convolutional prior",
     "two conv3x3 stages with pooling, flattened spatial head", "3 full", "0--19", 60),
]
t3 = []
for fid, name, rq, purpose, models, conds, seeds, n in families:
    t3.append(f"{fid} & {name} & {rq} & {purpose} & {models} & {conds} & {seeds} & {thousands(n)} \\\\")
write(OUT / "table3_families.tex", "\n".join(t3) + "\n")
assert sum(fam_counts.values()) == 1615

# ----------------------------------------------------------------------------------------------- Table 4
T4_GROUPS = [
    ("Branched models", [
        ("main_grid", "dann_lrf", "\\DANNLRF{}"),
        ("main_grid", "dann_random", "\\DANNRANDOM{}"),
        ("main_grid", "dann_grf", "\\DANNGRF{}"),
        ("F6_channel", "dann_lrf_channel", "\\DANNLRF{}, channel-aware"),
        ("main_grid", "naive_branch", "\\NAIVEBRANCH{}"),
        ("F8_fairness", "naive_branch_matched_init", "\\NAIVEBRANCH{}, matched init."),
    ]),
    ("Dense networks", [
        ("main_grid", "mlp_param", "\\MLPPARAM{}"),
        ("F2_baselines", "mlp_matched", "MLP-Matched"),
        ("main_grid", "vann_same", "\\VANNSAME{}"),
    ]),
    ("Reference priors at the DANN budget", [
        ("F2_baselines", "lc_net", "LC network"),
        ("F2_baselines", "sparse_mlp", "Random-sparse MLP"),
        ("F2_baselines", "compact_cnn", "Compact CNN"),
        ("arm", "cnn_flat", "Spatial-head CNN$^{\\dagger}$"),
    ]),
    ("Convolutional stem with three heads", [
        ("F9_convstem", "stem_dann_lrf", "Stem + \\DANNLRF{} head"),
        ("F9_convstem", "stem_naive_branch", "Stem + \\NAIVEBRANCH{} head"),
        ("F9_convstem", "stem_mlp", "Stem + MLP head"),
    ]),
]
ARM_DATASET = {"fashion_full": "fashionmnist", "kmnist_full": "kmnist", "cifar_full": "cifar10"}
t4 = []
t4_values: dict[str, dict[str, list[float]]] = {}
for gi, (group, rows) in enumerate(T4_GROUPS):
    if gi:
        t4.append("\\midrule")
    t4.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{group}}}}} \\\\")
    for family, model, label in rows:
        p28 = params.get((model, "fashionmnist"))
        p32 = params.get((model, "cifar10"))
        cells = []
        for cond in CONDITIONS:
            value = None
            if family == "arm":
                if cond in ARM_DATASET:
                    r = arm[(arm.model == "cnn_flat") & (arm.dataset == ARM_DATASET[cond])]
                    assert len(r) == 1
                    value = (float(r.iloc[0].acc_mean_pp), float(r.iloc[0].acc_sd_pp), int(r.iloc[0].n))
            else:
                rows_ = summary[(summary.family == family) & (summary.condition == cond) & (summary.model == model)]
                if len(rows_):
                    value = cell(family, cond, model)
            if value is None:
                cells.append("$\\cdot$")
            else:
                assert value[2] == 20, (family, model, cond, value[2])
                cells.append(pm(value[0], value[1]))
                t4_values.setdefault(model, {})[cond] = [round(value[0], 4), round(value[1], 4)]
        t4.append(f"{label} & {thousands(p28) if p28 else '$\\cdot$'} & {thousands(p32) if p32 else '$\\cdot$'} & "
                  + " & ".join(cells) + " \\\\")
write(OUT / "table4_accuracy.tex", "\n".join(t4) + "\n")
NUMBERS["table4"] = t4_values

# ----------------------------------------------------------------------------------------------- Table 5 / 6
dec = decision.set_index("criterion_id")
USED_IDS: list[str] = []


def drow(cid: str, label: str, cond_label: str) -> str:
    r = dec.loc[cid]
    USED_IDS.append(cid)
    return (f"{label} & {cond_label} & {fmt(r.mean_diff_pp)} & {ci(r.ci_low_pp, r.ci_high_pp)} & "
            f"{pval(r.holm_p)} & {fmt(r.d_z)} & {fmt(r.rank_biserial)} & {LABELS[r.outcome]} \\\\")


NB = "\\DANNLRF{} $-$ \\NAIVEBRANCH{}"
t5 = ["\\multicolumn{8}{@{}l}{\\textit{Dendrite-level nonlinearity (RQ1)}} \\\\"]
for cond in CONDITIONS:
    t5.append(drow(f"F1:{cond}:dann_lrf-naive_branch", NB, COND_LABEL[cond]))
for cond in ("fashion_full", "kmnist_full", "cifar_full"):
    t5.append(drow(f"F8b:{cond}:dann_lrf-naive_branch_matched_init", NB + ", matched init.", COND_LABEL[cond]))
for cond in ("fashion_full", "cifar_full"):
    t5.append(drow(f"F9:{cond}:stem_dann_lrf-stem_naive_branch", "Stem: \\DANNLRF{} head $-$ \\NAIVEBRANCH{} head",
                   COND_LABEL[cond]))
t5.append("\\midrule")
t5.append("\\multicolumn{8}{@{}l}{\\textit{Routing prior, locality and the dense baseline (RQ2)}} \\\\")
for cond in CONDITIONS:
    t5.append(drow(f"F1:{cond}:dann_lrf-mlp_param", "\\DANNLRF{} $-$ \\MLPPARAM{}", COND_LABEL[cond]))
for cond in ("fashion_full", "kmnist_full", "cifar_full"):
    t5.append(drow(f"F1:{cond}:dann_lrf-dann_random", "\\DANNLRF{} $-$ \\DANNRANDOM{}", COND_LABEL[cond]))
t5.append(drow("F5:fashion:did_lrf-random", "Permutation effect on \\DANNLRF{} $-$ \\DANNRANDOM{}",
               "FashionMNIST full"))
t5.append(drow("F5:cifar:did_lrf-random", "Permutation effect on \\DANNLRF{} $-$ \\DANNRANDOM{}", "CIFAR-10 full"))
t5.append(drow("F5:fashion:mlp_param_permuted-unpermuted", "\\MLPPARAM{}, permuted $-$ unpermuted",
               "FashionMNIST full"))
t5.append(drow("F5:cifar:mlp_param_permuted-unpermuted", "\\MLPPARAM{}, permuted $-$ unpermuted", "CIFAR-10 full"))
t5.append(drow("F6:cifar_full:dann_lrf_channel-dann_lrf", "Channel-aware $-$ cross-channel \\DANNLRF{}",
               "CIFAR-10 full"))
t5.append(drow("F6:cifar_low02:dann_lrf_channel-dann_lrf", "Channel-aware $-$ cross-channel \\DANNLRF{}",
               "CIFAR-10 0.2"))
for cond in ("fashion_full", "cifar_full"):
    t5.append(drow(f"F9:{cond}:stem_dann_lrf-stem_mlp", "Stem: \\DANNLRF{} head $-$ MLP head", COND_LABEL[cond]))
write(OUT / "table5_contrasts.tex", "\n".join(t5) + "\n")

t6 = ["\\multicolumn{8}{@{}l}{\\textit{Pre-specified references (Holm over 11 contrasts)}} \\\\"]
for ref, ref_label in (("compact_cnn", "compact CNN"), ("lc_net", "LC network"), ("sparse_mlp", "random-sparse MLP")):
    for cond in ("fashion_full", "kmnist_full", "cifar_full"):
        t6.append(drow(f"F2:{cond}:dann_lrf-{ref}", f"\\DANNLRF{{}} $-$ {ref_label}", COND_LABEL[cond]))
for cond in ("cifar_full", "cifar_low02"):
    t6.append(drow(f"F2:{cond}:dann_lrf-mlp_matched", "\\DANNLRF{} $-$ MLP-Matched", COND_LABEL[cond]))
t6.append("\\midrule")
t6.append("\\multicolumn{8}{@{}l}{\\textit{Exploratory spatial-head CNN$^{\\dagger}$ (no test, no label)}} \\\\")
ARM_COND = {"fashionmnist": "FashionMNIST full", "kmnist": "KMNIST full", "cifar10": "CIFAR-10 full"}
for ref, ref_label in (("dann_lrf", "\\DANNLRF{}"), ("compact_cnn", "compact CNN")):
    for ds in ("fashionmnist", "kmnist", "cifar10"):
        r = arm_paired[(arm_paired.dataset == ds) & (arm_paired.reference == ref)]
        assert len(r) == 1
        r = r.iloc[0]
        t6.append(f"Spatial-head CNN$^{{\\dagger}}$ $-$ {ref_label} & {ARM_COND[ds]} & {fmt(r.mean_diff_pp)} & "
                  f"{ci(r.ci95_low_pp, r.ci95_high_pp)} & n/a & {fmt(r.d_z)} & {fmt(r.rank_biserial)} & exploratory \\\\")
write(OUT / "table6_references.tex", "\n".join(t6) + "\n")
assert sorted(USED_IDS) == sorted(decision.criterion_id), "Tables 5-6 must carry every inferential contrast once"

# ----------------------------------------------------------------------------------------------- Table 7
T7_MODELS = [("dann_lrf", "\\DANNLRF{}"), ("naive_branch", "\\NAIVEBRANCH{}"), ("mlp_param", "\\MLPPARAM{}"),
             ("mlp_matched", "MLP-Matched"), ("vann_same", "\\VANNSAME{}"), ("lc_net", "LC network"),
             ("sparse_mlp", "Random-sparse MLP$^{\\ddagger}$"), ("compact_cnn", "Compact CNN"),
             ("cnn_flat", "Spatial-head CNN$^{\\dagger}$")]
t7 = []
for di, ds in enumerate(("fashionmnist", "cifar10")):
    if di:
        t7.append("\\midrule")
    t7.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{DATASET_LABEL[ds]}}}}} \\\\")
    for model, label in T7_MODELS:
        rows = f10[(f10.dataset == ds) & (f10.model == model)]
        if not len(rows):
            continue
        cpu = rows[rows.device == "cpu"].iloc[0]
        gpu = rows[rows.device == "cuda"].iloc[0]
        assert int(cpu.trainable_params_effective) == int(gpu.trainable_params_effective)
        assert int(cpu.macs_per_image_effective) == int(gpu.macs_per_image_effective)
        t7.append(f"{label} & {thousands(cpu.trainable_params_effective)} & {thousands(cpu.macs_per_image_effective)} & "
                  f"{cpu.train_s_per_epoch_mean:.3f} ({cpu.train_s_per_epoch_sd:.3f}) & "
                  f"{cpu.inference_ms_per_1000_mean:.2f} ({cpu.inference_ms_per_1000_sd:.2f}) & "
                  f"{gpu.train_s_per_epoch_mean:.3f} ({gpu.train_s_per_epoch_sd:.3f}) & "
                  f"{gpu.inference_ms_per_1000_mean:.2f} ({gpu.inference_ms_per_1000_sd:.2f}) & "
                  f"{gpu.peak_cuda_train_allocated_MB_mean:.1f} \\\\")
write(OUT / "table7_timing.tex", "\n".join(t7) + "\n")
NUMBERS["f10_ratios"] = f10_ratios.to_dict(orient="records")
NUMBERS["f10_rss_mb"] = {f"{r.device}:{r.dataset}:{r.model}": round(float(r.host_peak_rss_MB_mean), 1)
                         for r in f10.itertuples()}
NUMBERS["f10_sd_rel_max_pct"] = round(float(max((f10.train_s_per_epoch_sd / f10.train_s_per_epoch_mean).max(),
                                                (f10.inference_ms_per_1000_sd / f10.inference_ms_per_1000_mean).max())
                                            * 100), 2)
NUMBERS["f10_foreign_load_max"] = {d: round(float(f10[f10.device == d].foreign_load_pinned_cores_max.max()), 4)
                                   for d in ("cpu", "cuda")}

# ----------------------------------------------------------------------------------------------- F8(c) descriptive
INPUTS.append(FROZEN_ARCHIVE)
sel_units = units[((units.family == "main_grid") & units.condition.isin(["fashion_full", "kmnist_full", "cifar_full"])
                   & units.model.isin(["dann_lrf", "naive_branch"]))
                  | ((units.family == "F8_fairness") & (units.model == "naive_branch_matched_init"))]
assert len(sel_units) == 180, len(sel_units)
want = {r.unit_id: r for r in sel_units.itertuples()}
histories: dict[str, bytes] = {}
with tarfile.open(FROZEN_ARCHIVE, "r:gz") as tf:
    for member in tf:
        if not member.name.endswith("/history.csv"):
            continue
        parts = member.name.split("/")
        uid = parts[2] if len(parts) > 3 and parts[1] == "units" else None
        if uid in want:
            data = tf.extractfile(member).read()
            if hashlib.sha256(data).hexdigest().upper() == want[uid].history_sha256.upper():
                histories[uid] = data
assert set(histories) == set(want), sorted(set(want) - set(histories))[:5]
diag_rows = []
for uid, raw in histories.items():
    u = want[uid]
    h = pd.read_csv(io.BytesIO(raw))
    sel = int(h.loc[h.val_loss.idxmin(), "epoch"])  # idxmin returns the first minimum
    assert sel == int(u.best_val_epoch), (uid, sel, u.best_val_epoch)
    test_correct = int(round(float(h.loc[h.epoch == sel, "test_acc"].iloc[0]) * 10000))
    assert test_correct == int(u.test_correct_at_best_val), uid
    last = h[h.epoch == h.epoch.max()].iloc[0]
    assert int(last.epoch) == 30
    diag_rows.append({"unit_id": uid, "condition": u.condition, "model": u.model, "seed": int(u.seed),
                      "selected_epoch": sel, "selected_last": sel == 30, "train_acc_final": float(last.train_acc),
                      "train_loss_final": float(last.train_loss), "val_loss_min": float(h.val_loss.min())})
diag = pd.DataFrame(diag_rows)
agg = (diag.groupby(["condition", "model"]).agg(n=("unit_id", "count"), selected_epoch_mean=("selected_epoch", "mean"),
                                                 selected_last_count=("selected_last", "sum"),
                                                 train_acc_final_mean=("train_acc_final", "mean"),
                                                 train_loss_final_mean=("train_loss_final", "mean"),
                                                 val_loss_min_mean=("val_loss_min", "mean")).reset_index())
agg.to_csv(OUT / "f8c_fit_diagnostics.csv", index=False)
OUTPUTS.append(OUT / "f8c_fit_diagnostics.csv")
diag.sort_values("unit_id").to_csv(OUT / "f8c_fit_diagnostics_units.csv", index=False)
OUTPUTS.append(OUT / "f8c_fit_diagnostics_units.csv")

# ----------------------------------------------------------------------------------------------- figures
label_font = FontProperties(fname=str(FONT_LABEL))
text_font = FontProperties(fname=str(FONT_TEXT))
assert "IBM Plex Sans" in label_font.get_name(), label_font.get_name()
assert "Barlow" in text_font.get_name(), text_font.get_name()
NUMBERS["fonts_loaded"] = {"label": label_font.get_name(), "text": text_font.get_name()}
matplotlib.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                            "axes.edgecolor": PALETTE["ink"], "axes.linewidth": 0.6,
                            "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"]})


def fp(base: FontProperties, size: float) -> FontProperties:
    f = base.copy()
    f.set_size(size)
    return f


def style_axes(ax, size: float = 7.5) -> None:
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(fp(label_font, size))
    ax.tick_params(width=0.6, length=2.5)


def panel_label(ax, letter: str) -> None:
    ax.text(0.5, -0.30, f"({letter})", transform=ax.transAxes, ha="center", va="top",
            fontproperties=fp(text_font, 9.0), color=PALETTE["ink"])


seq = LinearSegmentedColormap.from_list("slate_seq", [PALETTE["pale"], PALETTE["light"], PALETTE["slate"],
                                                      PALETTE["navy"]])
div = LinearSegmentedColormap.from_list("slate_amber", [PALETTE["amber"], "#F4F1EC", PALETTE["navy"]])
KS, BS = [8, 16, 32], [2, 4, 8]


def grid(condition_prefix: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    m = np.zeros((3, 3))
    s = np.zeros((3, 3))
    for i, k in enumerate(KS):
        for j, b in enumerate(BS):
            mean, sd, n = cell("F3_sensitivity", f"{condition_prefix}_K{k}_B{b}", model)
            assert n == 5
            m[i, j], s[i, j] = mean, sd
            prm = params_f3[(condition_prefix, k, b)]
            assert prm == 128 * b * (k + 1) + 128 * (b + 1) + 129 * 10, (k, b, prm)
    return m, s


params_f3: dict[tuple[str, int, int], int] = {}
for r in units[units.family == "F3_sensitivity"].itertuples():
    cond = r.condition
    if "_K" in cond:
        prefix, kb = cond.split("_K")
        k, b = kb.split("_B")
        params_f3[(prefix, int(k), int(b))] = int(r.effective_params)

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
plt.subplots_adjust(wspace=0.42, hspace=0.62)
fig_data = []
for ax, prefix, letter, title_note in ((axes[0, 0], "fashion_full", "a", "FashionMNIST"),
                                       (axes[0, 1], "kmnist_full", "b", "KMNIST")):
    m, s = grid(prefix, "dann_lrf")
    im = ax.imshow(m, cmap=seq, aspect="auto", vmin=m.min() - 1.5, vmax=m.max() + 0.3)
    for i in range(3):
        for j in range(3):
            dark = m[i, j] > (m.min() + 0.55 * (m.max() - m.min()))
            colour = "white" if dark else PALETTE["ink"]
            ax.text(j, i - 0.12, f"{m[i, j]:.2f}", ha="center", va="center", color=colour,
                    fontproperties=fp(label_font, 7.8))
            ax.text(j, i + 0.22, f"±{s[i, j]:.2f}", ha="center", va="center", color=colour,
                    fontproperties=fp(text_font, 6.9))
            fig_data.append({"panel": letter, "dataset": prefix, "K": KS[i], "B": BS[j], "acc_mean_pct": m[i, j],
                             "acc_sd_pct": s[i, j], "params": params_f3[(prefix, KS[i], BS[j])]})
    ax.set_xticks(range(3), [str(b) for b in BS])
    ax.set_yticks(range(3), [str(k) for k in KS])
    ax.set_xlabel(f"Branches per soma, B ({title_note})", fontproperties=fp(label_font, 8.0))
    ax.set_ylabel("Features per dendrite, K", fontproperties=fp(label_font, 8.0))
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Accuracy (%)", fontproperties=fp(label_font, 7.5))
    for t in cb.ax.get_yticklabels():
        t.set_fontproperties(fp(label_font, 7.0))
    cb.outline.set_linewidth(0.5)
    style_axes(ax)
    panel_label(ax, letter)

ax = axes[1, 0]
dm = np.zeros((3, 3))
dl = np.zeros((3, 3))
dh = np.zeros((3, 3))
for i, k in enumerate(KS):
    for j, b in enumerate(BS):
        r = descriptive[descriptive.id == f"F3:kmnist_full_K{k}_B{b}:dann_lrf-dann_random"]
        assert len(r) == 1
        dm[i, j], dl[i, j], dh[i, j] = float(r.iloc[0].mean_diff_pp), float(r.iloc[0].ci95_low_pp), float(
            r.iloc[0].ci95_high_pp)
lim = float(np.abs(dm).max()) * 1.05
im = ax.imshow(dm, cmap=div, norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), aspect="auto")
for i in range(3):
    for j in range(3):
        colour = "white" if abs(dm[i, j]) > 0.55 * lim else PALETTE["ink"]
        ax.text(j, i - 0.14, fmt(dm[i, j]).replace("$-$", "\u2212"), ha="center", va="center", color=colour,
                fontproperties=fp(label_font, 7.8))
        ax.text(j, i + 0.22, f"[{dl[i, j]:.2f}, {dh[i, j]:.2f}]".replace("-", "\u2212"), ha="center", va="center",
                color=colour, fontproperties=fp(text_font, 6.4))
        fig_data.append({"panel": "c", "dataset": "kmnist_full", "K": KS[i], "B": BS[j],
                         "lrf_minus_random_pp": dm[i, j], "ci_low_pp": dl[i, j], "ci_high_pp": dh[i, j]})
ax.set_xticks(range(3), [str(b) for b in BS])
ax.set_yticks(range(3), [str(k) for k in KS])
ax.set_xlabel("Branches per soma, B (KMNIST)", fontproperties=fp(label_font, 8.0))
ax.set_ylabel("Features per dendrite, K", fontproperties=fp(label_font, 8.0))
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("DANN-LRF \u2212 DANN-RANDOM (pp)", fontproperties=fp(label_font, 7.0))
for t in cb.ax.get_yticklabels():
    t.set_fontproperties(fp(label_font, 7.0))
cb.outline.set_linewidth(0.5)
style_axes(ax)
panel_label(ax, "c")

ax = axes[1, 1]
patch_sides = [2, 4, 6]
for prefix, colour, marker, name, dx in (("fashion_full", PALETTE["navy"], "o", "FashionMNIST", -0.06),
                                         ("kmnist_full", PALETTE["slate"], "s", "KMNIST", 0.06)):
    means, sds = [], []
    for side in patch_sides:
        cond = f"{prefix}_K16_B4" if side == 4 else f"{prefix}_P{side}"
        mean, sd, n = cell("F3_sensitivity", cond, "dann_lrf")
        assert n == 5
        means.append(mean)
        sds.append(sd)
        fig_data.append({"panel": "d", "dataset": prefix, "patch": side, "acc_mean_pct": mean, "acc_sd_pct": sd})
    xs = np.array(patch_sides, dtype=float) + dx
    ax.errorbar(xs, means, yerr=sds, color=colour, marker=marker, markersize=4.2, linewidth=1.1, capsize=2.2,
                elinewidth=0.8, label=name, markeredgecolor=PALETTE["ink"], markeredgewidth=0.4)
ax.set_xticks(patch_sides, [f"{p}\u00d7{p}" for p in patch_sides])
ax.set_xlabel("Patch size of the local field", fontproperties=fp(label_font, 8.0))
ax.set_ylabel("Accuracy (%)", fontproperties=fp(label_font, 8.0))
ax.grid(axis="y", color=PALETTE["pale"], linewidth=0.6)
leg = ax.legend(prop=fp(text_font, 7.5), frameon=False, loc="lower right")
style_axes(ax)
panel_label(ax, "d")
pd.DataFrame(fig_data).to_csv(OUT / "figure_data_sensitivity.csv", index=False)
OUTPUTS.append(OUT / "figure_data_sensitivity.csv")
for target in (MS / "fig_sensitivity.png", OUT / "fig_sensitivity.pdf"):
    fig.savefig(target, dpi=600, bbox_inches="tight", pad_inches=0.03)
    OUTPUTS.append(target)
plt.close(fig)

# Figure 4: slope dose-response and learning-rate controls
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
plt.subplots_adjust(wspace=0.36, hspace=0.62)
slopes = [("a0", 0.0), ("a0p05", 0.05), ("a0p1", 0.1), ("a0p2", 0.2), ("a0p5", 0.5), ("a1", 1.0)]
fig_data = []
for ax, (cond, ds, letter) in zip((axes[0, 0], axes[0, 1], axes[1, 0]),
                                  (("fashion_full", "fashionmnist", "a"), ("kmnist_full", "kmnist", "b"),
                                   ("cifar_full", "cifar10", "c"))):
    means, sds = [], []
    for key, value in slopes:
        mean, sd, n = cell("F7_slope", cond, f"dann_lrf_slope_{key}")
        assert n == 10
        means.append(mean)
        sds.append(sd)
        fig_data.append({"panel": letter, "condition": cond, "alpha_d": value, "acc_mean_pct": mean,
                         "acc_sd_pct": sd, "n": n})
    xs = np.arange(len(slopes))
    ax.errorbar(xs, means, yerr=sds, color=PALETTE["navy"], marker="o", markersize=4.0, linewidth=1.1, capsize=2.2,
                elinewidth=0.8, markeredgecolor=PALETTE["ink"], markeredgewidth=0.4)
    ax.scatter([2], [means[2]], s=46, facecolors="none", edgecolors=PALETTE["amber"], linewidths=1.0, zorder=3)
    ax.set_xticks(xs, ["0", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_xlabel(f"Dendrite activation slope ({DATASET_LABEL[ds]})", fontproperties=fp(label_font, 8.0))
    ax.set_ylabel("Accuracy (%)", fontproperties=fp(label_font, 8.0))
    ax.grid(axis="y", color=PALETTE["pale"], linewidth=0.6)
    style_axes(ax)
    panel_label(ax, letter)

ax = axes[1, 1]
lrs = [("3em4", 3e-4), ("1em3", 1e-3), ("3em3", 3e-3)]
for ds_cond, colour, marker, name, factor in (("fashion_full", PALETTE["navy"], "o", "FashionMNIST", 0.93),
                                              ("kmnist_full", PALETTE["slate"], "s", "KMNIST", 1.0),
                                              ("cifar_full", PALETTE["light"], "^", "CIFAR-10", 1.07)):
    xs, ys, lo, hi = [], [], [], []
    for key, lr in lrs:
        r = descriptive[descriptive.id == f"F8a:{ds_cond}_lr{key}:dann_lrf-naive_branch"]
        assert len(r) == 1
        r = r.iloc[0]
        xs.append(lr * factor)
        ys.append(float(r.mean_diff_pp))
        lo.append(float(r.mean_diff_pp) - float(r.ci95_low_pp))
        hi.append(float(r.ci95_high_pp) - float(r.mean_diff_pp))
        fig_data.append({"panel": "d", "condition": ds_cond, "lr": lr, "lrf_minus_nb_pp": float(r.mean_diff_pp),
                         "ci_low_pp": float(r.ci95_low_pp), "ci_high_pp": float(r.ci95_high_pp), "n": int(r.n)})
    ax.errorbar(xs, ys, yerr=[lo, hi], color=colour, marker=marker, markersize=4.2, linewidth=1.0, capsize=2.2,
                elinewidth=0.8, label=name, markeredgecolor=PALETTE["ink"], markeredgewidth=0.4)
ax.axhline(0.0, color=PALETTE["ink"], linewidth=0.7, linestyle=(0, (3, 2)))
ax.set_xscale("log")
ax.set_xticks([3e-4, 1e-3, 3e-3], ["0.0003", "0.001", "0.003"])
ax.minorticks_off()
ax.set_ylim(-0.95, 2.15)
ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5], ["\u22120.5", "0", "0.5", "1.0", "1.5"])
ax.set_xlabel("Learning rate (Adam)", fontproperties=fp(label_font, 8.0))
ax.set_ylabel("DANN-LRF \u2212 Naive-Branch (pp)", fontproperties=fp(label_font, 8.0))
ax.grid(axis="y", color=PALETTE["pale"], linewidth=0.6)
ax.legend(prop=fp(text_font, 7.5), frameon=False, loc="upper center", ncol=3, columnspacing=1.0, handlelength=1.6,
          borderaxespad=0.2)
style_axes(ax)
panel_label(ax, "d")
pd.DataFrame(fig_data).to_csv(OUT / "figure_data_controls.csv", index=False)
OUTPUTS.append(OUT / "figure_data_controls.csv")
for target in (MS / "fig_controls.png", OUT / "fig_controls.pdf"):
    fig.savefig(target, dpi=600, bbox_inches="tight", pad_inches=0.03)
    OUTPUTS.append(target)
plt.close(fig)

# ----------------------------------------------------------------------------------------------- text numbers
NUMBERS["composite"] = composite.to_dict(orient="records")
NUMBERS["replication"] = {"rows": int(len(replication)), "sign_agrees": int(replication.sign_agrees.sum()),
                          "holm_decision_agrees": int(replication.holm_decision_agrees.sum())}
NUMBERS["f4_sd_pct"] = {r.id: {"sd_A": round(float(r.sd_A) * 100, 3), "sd_B": round(float(r.sd_B) * 100, 3),
                               "sd_diff": round(float(r.sd_A_minus_B) * 100, 3)}
                        for r in descriptive[descriptive.id.str.startswith("F4:")].itertuples()}
NUMBERS["f8c"] = agg.to_dict(orient="records")
write(OUT / "text_numbers.json", json.dumps(NUMBERS, indent=1, default=float) + "\n")

prov = {
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "script": {"path": str(Path(__file__).relative_to(P)).replace("\\", "/"), "sha256": sha(Path(__file__))},
    "python": platform.python_version(),
    "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    "fonts": {"label": [str(FONT_LABEL), sha(FONT_LABEL)], "text": [str(FONT_TEXT), sha(FONT_TEXT)]},
    "inputs": [{"path": str(p.relative_to(P)).replace("\\", "/"), "sha256": sha(p)} for p in INPUTS],
    "outputs": [{"path": str(p.relative_to(P)).replace("\\", "/"), "sha256": sha(p)} for p in OUTPUTS],
}
(OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"outputs": len(OUTPUTS), "inputs": len(INPUTS), "units_per_family": NUMBERS["units_per_family"],
                  "fonts": NUMBERS["fonts_loaded"]}, indent=1))
