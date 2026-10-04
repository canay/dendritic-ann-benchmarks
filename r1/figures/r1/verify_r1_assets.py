"""Independent second check of figures/r1/out (written separately from build_r1_assets.py; no pandas).

Every number in the generated LaTeX rows and figure-data files is re-read with the csv module from the verified
source files through an independently written label -> source mapping, and the protocol F8(c) diagnostics are
recomputed from the raw archive with plain Python. Any disagreement larger than the printed rounding is a failure.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

P = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / "figures" / "r1" / "out"  # argv: positive-control copy
FPO = P / "runs" / "2026-10-03_claude_mta_cuda_r1_frozen" / "processed_outputs"
ARCHIVE = P / "runs" / "2026-10-03_claude_mta_cuda_r1_frozen" / "raw_outputs" / \
    "2026-10-03_claude_mta_cuda_r1_frozen__public_outputs.tar.gz"  # public copy
F10 = P / "runs" / "2026-10-04_claude_mta_cuda_r1_f10_gpu" / "processed_outputs" / "f10_summary.csv"
ARM = P / "runs" / "2026-10-04_claude_mta_cuda_r1_cnnflat" / "processed_outputs"
FAIL: list[str] = []
CHECKED = {"cells": 0}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(token: str) -> float:
    token = token.strip().replace("$-$", "-").replace("$<$", "<")
    return float(token)


def close(a: float, b: float, d: int, what: str) -> None:
    CHECKED["cells"] += 1
    if abs(a - b) > 0.5 * 10 ** (-d) + 1e-9:
        FAIL.append(f"{what}: table {a} vs source {b}")


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.rstrip().rstrip("\\").split(" & ")]


# ------------------------------------------------------------------ Table 4
summary = rows(FPO / "summary_by_cell.csv")
arm_summary = rows(ARM / "arm_summary.csv")
LABEL_TO_SRC = {
    "\\DANNLRF{}": ("main_grid", "dann_lrf"), "\\DANNRANDOM{}": ("main_grid", "dann_random"),
    "\\DANNGRF{}": ("main_grid", "dann_grf"), "\\DANNLRF{}, channel-aware": ("F6_channel", "dann_lrf_channel"),
    "\\NAIVEBRANCH{}": ("main_grid", "naive_branch"),
    "\\NAIVEBRANCH{}, matched init.": ("F8_fairness", "naive_branch_matched_init"),
    "\\MLPPARAM{}": ("main_grid", "mlp_param"), "MLP-Matched": ("F2_baselines", "mlp_matched"),
    "\\VANNSAME{}": ("main_grid", "vann_same"), "LC network": ("F2_baselines", "lc_net"),
    "Random-sparse MLP": ("F2_baselines", "sparse_mlp"), "Compact CNN": ("F2_baselines", "compact_cnn"),
    "Spatial-head CNN$^{\\dagger}$": ("arm", "cnn_flat"),
    "Stem + \\DANNLRF{} head": ("F9_convstem", "stem_dann_lrf"),
    "Stem + \\NAIVEBRANCH{} head": ("F9_convstem", "stem_naive_branch"),
    "Stem + MLP head": ("F9_convstem", "stem_mlp"),
}
CONDS = ["fashion_full", "kmnist_full", "cifar_full", "fashion_low02", "fashion_low01", "cifar_low02"]
ARM_DS = {"fashion_full": "fashionmnist", "kmnist_full": "kmnist", "cifar_full": "cifar10"}
seen_models = set()
for line in (OUT / "table4_accuracy.tex").read_text(encoding="utf-8").splitlines():
    if line.startswith("\\multicolumn") or line.startswith("\\midrule") or not line.strip():
        continue
    cells = split_row(line)
    assert len(cells) == 9, line
    fam, model = LABEL_TO_SRC[cells[0]]
    seen_models.add(model)
    for cond, text in zip(CONDS, cells[3:]):
        if fam == "arm":
            src = [r for r in arm_summary if r["model"] == "cnn_flat" and r["dataset"] == ARM_DS.get(cond, "-")]
            vals = [(float(r["acc_mean_pp"]), float(r["acc_sd_pp"]), int(r["n"])) for r in src]
        else:
            src = [r for r in summary if r["family"] == fam and r["condition"] == cond and r["model"] == model]
            vals = [(float(r["acc_mean"]) * 100, float(r["acc_sd"]) * 100, int(r["n_seeds"])) for r in src]
        if text == "$\\cdot$":
            if vals:
                FAIL.append(f"T4 {model} {cond}: source exists but table shows a dot")
            continue
        assert len(vals) == 1, (model, cond)
        m, s = text.split("$\\pm$")
        close(num(m), vals[0][0], 2, f"T4 {model} {cond} mean")
        close(num(s), vals[0][1], 2, f"T4 {model} {cond} sd")
        if vals[0][2] != 20:
            FAIL.append(f"T4 {model} {cond}: n={vals[0][2]}")
if len(seen_models) != 16:
    FAIL.append(f"T4 has {len(seen_models)} models")

# ------------------------------------------------------------------ Tables 5 and 6
decision = {r["criterion_id"]: r for r in rows(FPO / "decision_inputs.csv")}
COND_FROM_LABEL = {"FashionMNIST full": "fashion_full", "KMNIST full": "kmnist_full", "CIFAR-10 full": "cifar_full",
                   "FashionMNIST 0.2": "fashion_low02", "FashionMNIST 0.1": "fashion_low01",
                   "CIFAR-10 0.2": "cifar_low02"}


def cid_for(label: str, cond: str) -> str:
    short = {"fashion_full": "fashion", "cifar_full": "cifar"}
    table = {
        "\\DANNLRF{} $-$ \\NAIVEBRANCH{}": f"F1:{cond}:dann_lrf-naive_branch",
        "\\DANNLRF{} $-$ \\NAIVEBRANCH{}, matched init.": f"F8b:{cond}:dann_lrf-naive_branch_matched_init",
        "Stem: \\DANNLRF{} head $-$ \\NAIVEBRANCH{} head": f"F9:{cond}:stem_dann_lrf-stem_naive_branch",
        "\\DANNLRF{} $-$ \\MLPPARAM{}": f"F1:{cond}:dann_lrf-mlp_param",
        "\\DANNLRF{} $-$ \\DANNRANDOM{}": f"F1:{cond}:dann_lrf-dann_random",
        "Permutation effect on \\DANNLRF{} $-$ \\DANNRANDOM{}": f"F5:{short.get(cond, cond)}:did_lrf-random",
        "\\MLPPARAM{}, permuted $-$ unpermuted": f"F5:{short.get(cond, cond)}:mlp_param_permuted-unpermuted",
        "Channel-aware $-$ cross-channel \\DANNLRF{}": f"F6:{cond}:dann_lrf_channel-dann_lrf",
        "Stem: \\DANNLRF{} head $-$ MLP head": f"F9:{cond}:stem_dann_lrf-stem_mlp",
        "\\DANNLRF{} $-$ compact CNN": f"F2:{cond}:dann_lrf-compact_cnn",
        "\\DANNLRF{} $-$ LC network": f"F2:{cond}:dann_lrf-lc_net",
        "\\DANNLRF{} $-$ random-sparse MLP": f"F2:{cond}:dann_lrf-sparse_mlp",
        "\\DANNLRF{} $-$ MLP-Matched": f"F2:{cond}:dann_lrf-mlp_matched",
    }
    return table[label]


OUTCOME = {"supported": "supported", "not supported": "not_supported", "reverse": "reverse"}
used = []
arm_paired = rows(ARM / "arm_paired.csv")
for name in ("table5_contrasts.tex", "table6_references.tex"):
    for line in (OUT / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("\\multicolumn") or line.startswith("\\midrule") or not line.strip():
            continue
        c = split_row(line)
        assert len(c) == 8, line
        cond = COND_FROM_LABEL[c[1]]
        lo, hi = [num(x) for x in c[3].strip("[]").split(",")]
        if c[7] == "exploratory":
            ref = "dann_lrf" if c[0].endswith("\\DANNLRF{}") else "compact_cnn"
            src = [r for r in arm_paired if r["dataset"] == ARM_DS[cond] and r["reference"] == ref]
            assert len(src) == 1
            r = src[0]
            close(num(c[2]), float(r["mean_diff_pp"]), 2, f"{name} arm {cond} {ref} mean")
            close(lo, float(r["ci95_low_pp"]), 2, f"{name} arm {cond} {ref} lo")
            close(hi, float(r["ci95_high_pp"]), 2, f"{name} arm {cond} {ref} hi")
            close(num(c[5]), float(r["d_z"]), 2, f"{name} arm d_z")
            close(num(c[6]), float(r["rank_biserial"]), 2, f"{name} arm r_rb")
            continue
        cid = cid_for(c[0], cond)
        used.append(cid)
        r = decision[cid]
        close(num(c[2]), float(r["mean_diff_pp"]), 2, f"{cid} mean")
        close(lo, float(r["ci_low_pp"]), 2, f"{cid} lo")
        close(hi, float(r["ci_high_pp"]), 2, f"{cid} hi")
        p = float(r["holm_p"])
        if c[4] == "$<$0.001":
            if not p < 0.001:
                FAIL.append(f"{cid} p shown <0.001 but is {p}")
        else:
            close(float(c[4]), p, 3, f"{cid} holm p")
        close(num(c[5]), float(r["d_z"]), 2, f"{cid} d_z")
        close(num(c[6]), float(r["rank_biserial"]), 2, f"{cid} r_rb")
        if OUTCOME[c[7]] != r["outcome"]:
            FAIL.append(f"{cid} label {c[7]} vs {r['outcome']}")
if sorted(used) != sorted(decision):
    FAIL.append(f"Tables 5-6 carry {len(used)} contrasts; source has {len(decision)}")

# ------------------------------------------------------------------ Table 7
f10 = rows(F10)
T7 = {"\\DANNLRF{}": "dann_lrf", "\\NAIVEBRANCH{}": "naive_branch", "\\MLPPARAM{}": "mlp_param",
      "MLP-Matched": "mlp_matched", "\\VANNSAME{}": "vann_same", "LC network": "lc_net",
      "Random-sparse MLP$^{\\ddagger}$": "sparse_mlp", "Compact CNN": "compact_cnn",
      "Spatial-head CNN$^{\\dagger}$": "cnn_flat"}
dataset = None
pair = re.compile(r"([0-9.]+) \(([0-9.]+)\)")
for line in (OUT / "table7_timing.tex").read_text(encoding="utf-8").splitlines():
    if "FashionMNIST" in line and line.startswith("\\multicolumn"):
        dataset = "fashionmnist"
        continue
    if "CIFAR-10" in line and line.startswith("\\multicolumn"):
        dataset = "cifar10"
        continue
    if line.startswith("\\midrule") or not line.strip():
        continue
    c = split_row(line)
    model = T7[c[0]]
    cpu = [r for r in f10 if r["device"] == "cpu" and r["dataset"] == dataset and r["model"] == model][0]
    gpu = [r for r in f10 if r["device"] == "cuda" and r["dataset"] == dataset and r["model"] == model][0]
    if int(c[1].replace(",", "")) != int(cpu["trainable_params_effective"]):
        FAIL.append(f"T7 params {model} {dataset}")
    if int(c[2].replace(",", "")) != int(cpu["macs_per_image_effective"]):
        FAIL.append(f"T7 MACs {model} {dataset}")
    for text, src, key, d in ((c[3], cpu, "train_s_per_epoch", 3), (c[4], cpu, "inference_ms_per_1000", 2),
                              (c[5], gpu, "train_s_per_epoch", 3), (c[6], gpu, "inference_ms_per_1000", 2)):
        m = pair.fullmatch(text)
        assert m, text
        close(float(m.group(1)), float(src[key + "_mean"]), d, f"T7 {model} {dataset} {key} mean")
        close(float(m.group(2)), float(src[key + "_sd"]), d, f"T7 {model} {dataset} {key} sd")
    close(float(c[7]), float(gpu["peak_cuda_train_allocated_MB_mean"]), 1, f"T7 {model} {dataset} MB")

# ------------------------------------------------------------------ Table 3 counts
units = rows(FPO / "unit_table.csv")
counts: dict[str, int] = {}
for u in units:
    counts[u["family"]] = counts.get(u["family"], 0) + 1
FAM = {"F1": "main_grid", "F2": "F2_baselines", "F3": "F3_sensitivity", "F4": "F4_randomness", "F5": "F5_shuffled",
       "F6": "F6_channel", "F7": "F7_slope", "F8": "F8_fairness", "F9": "F9_convstem"}
for line in (OUT / "table3_families.tex").read_text(encoding="utf-8").splitlines():
    c = split_row(line)
    n = int(c[-1].replace(",", ""))
    if c[0] in FAM:
        if counts[FAM[c[0]]] != n:
            FAIL.append(f"T3 {c[0]} {n} vs {counts[FAM[c[0]]]}")
    elif c[0] == "F10":
        n_f10 = sum(int(r["seeds"]) for r in f10)
        if n_f10 != n:
            FAIL.append(f"T3 F10 {n} vs {n_f10}")
    elif c[0] == "A12":
        n_arm = sum(1 for r in rows(ARM / "arm_unit_table.csv") if r["source"] == "arm" and r["family"] != "MC002_anchor")
        if n_arm != n:
            FAIL.append(f"T3 A12 {n} vs {n_arm}")

# ------------------------------------------------------------------ F8(c) recomputed with plain Python
want = {}
for u in units:
    if (u["family"] == "main_grid" and u["condition"] in ("fashion_full", "kmnist_full", "cifar_full")
            and u["model"] in ("dann_lrf", "naive_branch")) or (u["family"] == "F8_fairness"
                                                              and u["model"] == "naive_branch_matched_init"):
        want[u["unit_id"]] = u
acc: dict[tuple[str, str], list[tuple[int, float, float, float]]] = {}
with tarfile.open(ARCHIVE, "r:gz") as tf:
    for member in tf:
        if not member.name.endswith("/history.csv"):
            continue
        uid = member.name.split("/")[2]
        if uid not in want:
            continue
        data = tf.extractfile(member).read()
        if hashlib.sha256(data).hexdigest().upper() != want[uid]["history_sha256"].upper():
            continue
        recs = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        best = None
        for r in recs:
            v = float(r["val_loss"])
            if best is None or v < best[0]:
                best = (v, int(r["epoch"]))
        final = max(recs, key=lambda r: int(r["epoch"]))
        u = want[uid]
        acc.setdefault((u["condition"], u["model"]), []).append(
            (best[1], float(final["train_acc"]), float(final["train_loss"]), best[0]))
diag = rows(OUT / "f8c_fit_diagnostics.csv")
for d in diag:
    vals = acc[(d["condition"], d["model"])]
    if len(vals) != int(d["n"]):
        FAIL.append(f"F8c n {d['condition']} {d['model']}")
    n = len(vals)
    checks = (("selected_epoch_mean", sum(v[0] for v in vals) / n),
              ("selected_last_count", sum(1 for v in vals if v[0] == 30)),
              ("train_acc_final_mean", sum(v[1] for v in vals) / n),
              ("train_loss_final_mean", sum(v[2] for v in vals) / n),
              ("val_loss_min_mean", sum(v[3] for v in vals) / n))
    for key, value in checks:
        if abs(float(d[key]) - value) > 1e-9:
            FAIL.append(f"F8c {d['condition']} {d['model']} {key}: {d[key]} vs {value}")
        CHECKED["cells"] += 1

# ------------------------------------------------------------------ figure data against sources
desc = {r["id"]: r for r in rows(FPO / "descriptive.csv")}
for r in rows(OUT / "figure_data_sensitivity.csv"):
    if r["panel"] in ("a", "b", "c"):
        k, b = int(float(r["K"])), int(float(r["B"]))  # pandas wrote the mixed-panel columns as floats
    if r["panel"] == "c":
        src = desc[f"F3:kmnist_full_K{k}_B{b}:dann_lrf-dann_random"]
        close(float(r["lrf_minus_random_pp"]), float(src["mean_diff_pp"]), 9, "fig3c")
    elif r["panel"] in ("a", "b"):
        src = [s for s in summary if s["family"] == "F3_sensitivity" and s["model"] == "dann_lrf"
               and s["condition"] == f"{r['dataset']}_K{k}_B{b}"][0]
        close(float(r["acc_mean_pct"]), float(src["acc_mean"]) * 100, 9, "fig3ab")
    else:
        side = int(float(r["patch"]))
        cond = f"{r['dataset']}_K16_B4" if side == 4 else f"{r['dataset']}_P{side}"
        src = [s for s in summary if s["family"] == "F3_sensitivity" and s["model"] == "dann_lrf"
               and s["condition"] == cond][0]
        close(float(r["acc_mean_pct"]), float(src["acc_mean"]) * 100, 9, "fig3d")
for r in rows(OUT / "figure_data_controls.csv"):
    if r["panel"] == "d":
        key = {0.0003: "3em4", 0.001: "1em3", 0.003: "3em3"}[float(r["lr"])]
        src = desc[f"F8a:{r['condition']}_lr{key}:dann_lrf-naive_branch"]
        close(float(r["lrf_minus_nb_pp"]), float(src["mean_diff_pp"]), 9, "fig4d")
    else:
        slope_key = {0.0: "a0", 0.05: "a0p05", 0.1: "a0p1", 0.2: "a0p2", 0.5: "a0p5", 1.0: "a1"}[float(r["alpha_d"])]
        src = [s for s in summary if s["family"] == "F7_slope" and s["condition"] == r["condition"]
               and s["model"] == f"dann_lrf_slope_{slope_key}"][0]
        close(float(r["acc_mean_pct"]), float(src["acc_mean"]) * 100, 9, "fig4abc")

# ------------------------------------------------------------------ provenance hashes
prov = json.loads((OUT / "PROVENANCE.json").read_text(encoding="utf-8"))
for item in prov["inputs"] + prov["outputs"]:
    path = P / item["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest().upper() != item["sha256"]:
        FAIL.append(f"provenance hash drift: {item['path']}")

print(json.dumps({"checked_values": CHECKED["cells"], "failures": len(FAIL)}, indent=1))
for f in FAIL[:40]:
    print("FAIL", f)
sys.exit(1 if FAIL else 0)
