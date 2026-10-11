"""Pre-specified analysis of the bounded tier D extension (MC-NEURO-R1-003; protocol Section 10, amendments A14-A16); no
model code. Written before any D1 or D2 unit existed.

    python -B r1/aggregate_tierd.py analyse --plan <plan_tierd_arm.json> --root <extracted run folder> --identity <IDENTITY.json>
        --frozen-plan <plan_frozen_all.json> --frozen-archive <frozen outputs.tar.gz> --frozen-receipt <frozen receipt.json>
        --frozen-identity <frozen IDENTITY.json> --out <dir>
    python -B r1/aggregate_tierd.py pilot --plan <plan_tierd_pilot.json> --root <pilot run folder> [--identity <IDENTITY.json>] --out <dir>
    python -B r1/aggregate_tierd.py selftest

analyse fails closed (exit 2, nothing written) when the plan is not the pre-specified unit set (a missing, extra or duplicate
cell or seed, a changed hyper-parameter), when the run folder holds a unit the plan does not name, when any planned unit is
missing or invalid (r1_runner.validate_result with the identity record), when an accuracy is not a whole number of the
10,000 test images, when a cell's parameter count differs from the pre-specified count, when the converted data tensors of a
dataset (their SHA-256, recorded by the runner) differ between units, or when the frozen archive does not hash to its receipt
or a frozen anchor reference is missing, invalid or not the anchor's own unit. Primary metric (A9): the
test accuracy at the validation-selected epoch, re-derived from history.csv (first epoch of minimum val_loss), as the integer
number of correct test images. Statistics: the frozen functions of r1/aggregate_r1.py (int_stats, holm, label): two-sided
Wilcoxon signed-rank on the seed-paired integer differences (exact when no difference is zero, no absolute difference is
tied and n <= 50; otherwise the normal approximation without continuity correction), Holm within each family (D1: four
contrasts; D2: three; the same contrasts of the 100-epoch budget arm of amendment A17, DEC-NEURO-020, in their own families
D1_e100 and D2_e100), 95 % percentile bootstrap of the mean difference with R = 10,000 and one fixed generator seed per
contrast, Cohen's d_z and the matched-pairs rank-biserial; labels by A8 (supported: Holm p < 0.05 and CI lower bound > 0;
reverse: Holm p < 0.05 and CI upper bound < 0; otherwise not supported; no difference when every paired difference is zero).
Composite rules: D1 transfer (the F9 rule: observed if both Naive-Branch-head contrasts are supported, dataset-specific if
exactly one, not observed otherwise; every reverse is named) and the D2 increment (the label of DANN-LRF minus Naive-Branch on
CIFAR-100), each at the primary 30-epoch protocol and in the 100-epoch arm, and the budget agreement (A17: the primary label is
the 30-epoch one; where the 100-epoch label of the same contrast differs, the statement is worded as budget-dependent and both
labels are reported). None enters the A8 two-of-three summary, which stays defined over FashionMNIST, KMNIST and CIFAR-10, and
none changes a NEURO-R1-FREEZE-001 verdict. Lineage anchors: each anchor history must equal its frozen unit's history byte
for byte; the within-run statistics do not depend on the anchors, a mismatch is a determinism finding and exits 3 after
writing the outputs.
Outputs: unit_table.csv, summary_by_cell.csv, contrasts.csv, decision_inputs.csv, composite_decisions.csv, anchors.csv,
ANALYSIS_RUN.json.

pilot (reference adequacy, STUDY_DESIGN 'Baselines And Comparators'; pilot_only, never evidence): per unit the final-epoch
training accuracy, the selected epoch and whether it is the last, the training-accuracy change over the last five epochs,
the test accuracy of the main model of the same dataset and seed (DANN-LRF head for D1, DANN-LRF for D2) and two flags:
selected at the last epoch with a still-rising training accuracy; final-epoch training accuracy below the main model's test
accuracy. Outputs: pilot_adequacy.csv, PILOT_RUN.json.

selftest: synthetic run folders and a synthetic frozen archive; checks the statistics against an independent recomputation
(exact Wilcoxon by dynamic programming, normal approximation, Holm, bootstrap, d_z, rank-biserial), the labels, the composite
rules, the anchor gate, the pilot flags and every fail-closed path. Exit 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import platform
import shutil
import statistics
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1.aggregate_r1 import BOOT_R, METRIC_COLUMNS, holm, int_stats, label, read_history, sha_file, write_csv  # noqa: E402
from r1.r1_runner import load_plan, unit_spec, unit_spec_sha256, validate_result  # noqa: E402

N_TEST = 10_000
SEEDS = list(range(20))
PILOT_SEED = 100
FAM_ANCHOR, FAM_D1, FAM_D2, FAM_PILOT = "MC003_anchor", "MC003_D1_tfstem", "MC003_D2_cifar100", "MC003_pilot"
D1_CONDITIONS = ("fashion_full", "cifar_full")
TF_HEADS = ("tf_stem_dann_lrf", "tf_stem_naive_branch", "tf_stem_mlp")
D2_CONDITION = "cifar100_full"
D2_MODELS = ("dann_lrf", "naive_branch", "mlp_param", "dann_random", "vann_same")
DATASET_OF = {"fashion_full": "fashionmnist", "cifar_full": "cifar10", "cifar100_full": "cifar100"}
# DEC-NEURO-020 (amendment A17): the pre-specified 100-epoch budget-sensitivity arm of D1 and D2 (same cells, seeds 0-19,
# every other hyper-parameter of Section 1 unchanged); its conditions carry the suffix _e100
FAM_D1_E100, FAM_D2_E100 = "MC003_D1_tfstem_e100", "MC003_D2_cifar100_e100"
E100 = "_e100"
BUDGET_EPOCHS = 100
D1_CONDITIONS_E100 = tuple(c + E100 for c in D1_CONDITIONS)
D2_CONDITION_E100 = D2_CONDITION + E100
DATASET_OF.update({c + E100: d for c, d in list(DATASET_OF.items())})
EPOCHS_OF_FAMILY = {FAM_D1_E100: BUDGET_EPOCHS, FAM_D2_E100: BUDGET_EPOCHS}  # every other family: the 30 epochs of BASE
# lineage anchors (condition, model) at seed 0 and the family of the frozen unit each one repeats
ANCHORS = [("cifar_full", m) for m in D2_MODELS] + [("fashion_full", "stem_dann_lrf")] + \
    [("cifar_full", m) for m in ("stem_dann_lrf", "stem_naive_branch", "stem_mlp")]
BASE = {"epochs": 30, "batch_size": 256, "lr": 0.001, "val_fraction": 0.1, "soma_units": 128, "branches_per_soma": 4,
        "sample_size": 16, "patch_h": 4, "patch_w": 4, "subset_fraction": 1.0}
# pre-specified trainable parameter counts (r1/check_tierd.py; amendments A14 and A15)
EXPECTED_PARAMS = {("fashion_full", "tf_stem_dann_lrf"): 13946, ("fashion_full", "tf_stem_naive_branch"): 13946,
                   ("fashion_full", "tf_stem_mlp"): 13839, ("cifar_full", "tf_stem_dann_lrf"): 14698,
                   ("cifar_full", "tf_stem_naive_branch"): 14698, ("cifar_full", "tf_stem_mlp"): 14534,
                   (D2_CONDITION, "dann_lrf"): 22244, (D2_CONDITION, "naive_branch"): 22244, (D2_CONDITION, "mlp_param"): 22367,
                   (D2_CONDITION, "dann_random"): 22244, (D2_CONDITION, "vann_same"): 1651940}
EXPECTED_PARAMS.update({(c + E100, m): v for (c, m), v in list(EXPECTED_PARAMS.items())})  # A17: same models, same counts
# contrasts A - B in a fixed order; bootstrap seed CONTRAST_SEED0 + k, k = position in this list
CONTRASTS = [(f"D1:{c}:tf_stem_dann_lrf-{b}", "D1", FAM_D1, c, "tf_stem_dann_lrf", b, "transfer" if b == "tf_stem_naive_branch" else "label")
             for c in D1_CONDITIONS for b in ("tf_stem_naive_branch", "tf_stem_mlp")] + \
            [(f"D2:{D2_CONDITION}:dann_lrf-{b}", "D2", FAM_D2, D2_CONDITION, "dann_lrf", b, "increment" if b == "naive_branch" else "label")
             for b in ("naive_branch", "mlp_param", "dann_random")]
# A17: the same seven contrasts at the 100-epoch budget, appended (k = 7-13), each in its own Holm family
CONTRASTS += [(f"D1_e100:{c}:tf_stem_dann_lrf-{b}", "D1_e100", FAM_D1_E100, c, "tf_stem_dann_lrf", b,
               "transfer" if b == "tf_stem_naive_branch" else "label") for c in D1_CONDITIONS_E100 for b in ("tf_stem_naive_branch", "tf_stem_mlp")] + \
             [(f"D2_e100:{D2_CONDITION_E100}:dann_lrf-{b}", "D2_e100", FAM_D2_E100, D2_CONDITION_E100, "dann_lrf", b,
               "increment" if b == "naive_branch" else "label") for b in ("naive_branch", "mlp_param", "dann_random")]
HOLM_GROUPS = ("D1", "D2", "D1_e100", "D2_e100")
PRIMARY_GROUPS = ("D1", "D2")


def primary_of(cid: str) -> str:
    """The 30-epoch counterpart of a 100-epoch contrast id (A17)."""
    return cid.replace("_e100", "")


# cells in a fixed order; bootstrap seed of the mean CELL_SEED0 + k
CELLS = [(FAM_D1, c, m) for c in D1_CONDITIONS for m in TF_HEADS] + [(FAM_D2, D2_CONDITION, m) for m in D2_MODELS]
CELLS += [(FAM_D1_E100, c, m) for c in D1_CONDITIONS_E100 for m in TF_HEADS] + [(FAM_D2_E100, D2_CONDITION_E100, m) for m in D2_MODELS]
CONTRAST_SEED0 = 20261004300
CELL_SEED0 = 20261004400
DIRECTION = "positive = A better than B (more test images correct)"
THRESHOLD = "holm_p < 0.05 and ci95_low > 0 (reverse: holm_p < 0.05 and ci95_high < 0)"
BOUNDARY = ("post-main-results extension (MC-NEURO-R1-003), reported whatever the outcome; outside the A8 two-of-three summary over "
            "FashionMNIST, KMNIST and CIFAR-10 (RQ2_robustness in R1_ANALYSIS_MANIFEST.json, RQ1 in the R1 manuscript numbering); "
            "changes no NEURO-R1-FREEZE-001 verdict")
PINNED = {"python": "3.12.12", "numpy": "2.3.5", "scipy": "1.18.0"}


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit_key(u: dict) -> tuple:
    return (u.get("family"), u.get("condition"), u.get("model"), u.get("seed"))


def expected_keys(kind: str) -> set:
    if kind == "pilot":
        return ({(FAM_PILOT, c, m, PILOT_SEED) for c in D1_CONDITIONS for m in TF_HEADS}
                | {(FAM_PILOT, D2_CONDITION, m, PILOT_SEED) for m in D2_MODELS})
    return ({(FAM_ANCHOR, c, m, 0) for c, m in ANCHORS} | {(FAM_D1, c, m, s) for c in D1_CONDITIONS for m in TF_HEADS for s in SEEDS}
            | {(FAM_D2, D2_CONDITION, m, s) for m in D2_MODELS for s in SEEDS}
            | {(FAM_D1_E100, c, m, s) for c in D1_CONDITIONS_E100 for m in TF_HEADS for s in SEEDS}
            | {(FAM_D2_E100, D2_CONDITION_E100, m, s) for m in D2_MODELS for s in SEEDS})


def plan_problems(units, kind: str) -> list:
    """Every deviation of the plan from the pre-specified unit set (A14-A16)."""
    problems, seen = [], set()
    for u in units:
        key, uid = unit_key(u), u.get("unit_id")
        if key in seen:
            problems.append(f"duplicate cell and seed {key}")
        seen.add(key)
        if not isinstance(u.get("seed"), int) or uid != f"{u.get('condition')}__{u.get('model')}__s{u['seed']:02d}":
            problems.append(f"{uid}: unit id does not follow condition__model__sNN")
        if u.get("dataset") != DATASET_OF.get(u.get("condition")):
            problems.append(f"{uid}: dataset {u.get('dataset')!r} does not belong to condition {u.get('condition')!r}")
        for k, v in BASE.items():
            want_v = EPOCHS_OF_FAMILY.get(u.get("family"), v) if k == "epochs" else v
            if u.get(k) != want_v:
                problems.append(f"{uid}: {k}={u.get(k)!r}, pre-specified {want_v!r}")
        r1_kind = str(u.get("model")) in TF_HEADS or str(u.get("model")).startswith("stem_")
        if (u.get("extra") or {}) != ({"r1_model": u.get("model")} if r1_kind else {}):
            problems.append(f"{uid}: extra {u.get('extra')!r} is not the pre-specified one")
    want = expected_keys(kind)
    problems += [f"missing from the plan: {k}" for k in sorted(want - seen, key=str)]
    problems += [f"not pre-specified: {k}" for k in sorted(seen - want, key=str)]
    return problems


def unit_metrics(unit_dir: Path) -> dict:
    r = read_json(unit_dir / "result.json")
    h = read_history(unit_dir / r["history_file"])
    best = min(h, key=lambda x: float(x["val_loss"]))  # first minimum in epoch order, as train_one_run selects
    acc = float(best["test_acc"])
    cnt = round(acc * N_TEST)
    if abs(acc * N_TEST - cnt) > 1e-6:
        raise ValueError(f"accuracy {acc!r} is not a whole number of the {N_TEST} test images")
    mi = r.get("model_info") or {}
    last = h[-1]
    return {"best_val_epoch": int(best["epoch"]), "test_correct": cnt, "test_acc": acc, "epochs": len(h),
            "train_acc_last_epoch": float(last["train_acc"]),
            "train_acc_change_last5": float(last["train_acc"]) - float(h[-6]["train_acc"]),
            "selected_last_epoch": int(best["epoch"]) == len(h),
            "effective_params": int(mi.get("effective_trainable_params", r["summary"]["trainable_params"])),
            "unit_seconds": (r.get("timing_seconds") or {}).get("unit_total"),
            "data_hashes": json.dumps(r.get("data_hashes"), sort_keys=True),
            "history_path": unit_dir / r["history_file"], "history_sha256": r["history_sha256"]}


def data_problems(metrics: dict) -> list:
    """One set of converted data tensors per dataset across every unit of the run (the runner records their SHA-256)."""
    seen = {}
    for (_f, _c, _m, _s), (u, m) in metrics.items():
        seen.setdefault(u["dataset"], set()).add(m["data_hashes"])
    out = [(ds, "no data hashes recorded") for ds, v in seen.items() if "null" in v]
    return out + [(ds, f"data tensors differ between units ({len(v)} hash sets)") for ds, v in seen.items() if len(v) > 1]


def collect(units, root: Path, identity, bad: list) -> dict:
    """Validated metrics per unit key; every problem is appended to `bad`."""
    out = {}
    for u in units:
        udir = root / "units" / u["unit_id"]
        reason = validate_result(u, udir, identity)
        if reason is not None:
            bad.append((u["unit_id"], reason))
            continue
        try:
            m = unit_metrics(udir)
        except (ValueError, KeyError, IndexError, OSError) as exc:
            bad.append((u["unit_id"], str(exc)))
            continue
        exp = EXPECTED_PARAMS.get((u["condition"], u["model"]))
        if u["family"] != FAM_ANCHOR and exp is not None and m["effective_params"] != exp:
            bad.append((u["unit_id"], f"{m['effective_params']} trainable parameters, pre-specified {exp}"))
            continue
        out[unit_key(u)] = (u, m)
    planned = {u["unit_id"] for u in units}
    units_dir = root / "units"
    if units_dir.is_dir():
        bad += [(p.name, "unit folder not named by the plan") for p in sorted(units_dir.iterdir()) if p.is_dir() and p.name not in planned]
    return out


def mean_ci(counts, seed: int):
    """95 % percentile bootstrap of the mean count (the int_stats resampling rule, R = 10,000)."""
    import numpy as np

    v = np.asarray([int(x) for x in counts], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    means = v[rng.integers(0, len(v), size=(BOOT_R, len(v)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def versions() -> dict:
    import numpy
    import scipy

    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}


def fail(bad, what: str) -> int:
    for uid, reason in bad[:20]:
        print("INVALID", uid, reason)
    print(f"FAIL: {len(bad)} problem(s) in {what}; nothing aggregated")
    return 2


# ---------------------------------------------------------------- analyse
def analyse(args) -> int:
    try:
        plan, frozen_plan = load_plan(Path(args.plan)), load_plan(Path(args.frozen_plan))
    except SystemExit as exc:
        return fail([("plan", str(exc))], "the plans")
    problems = plan_problems(plan["units"], "arm")
    if problems:
        return fail([("plan", p) for p in problems], "the plan")
    identity, frozen_ident, receipt = read_json(args.identity), read_json(args.frozen_identity), read_json(args.frozen_receipt)
    archive_sha = sha_file(Path(args.frozen_archive))
    if archive_sha != str(receipt["archive"]["sha256"]).upper():
        return fail([("frozen archive", "does not hash to its receipt")], "the frozen archive")

    bad = []
    metrics = collect(plan["units"], Path(args.root), identity, bad)
    if not bad:
        bad += data_problems(metrics)
    frozen_by_id = {u["unit_id"]: u for u in frozen_plan["units"]}
    wanted = {}
    for u in plan["units"]:
        if u["family"] != FAM_ANCHOR:
            continue
        f = frozen_by_id.get(u["unit_id"])
        fam = "F9_convstem" if u["model"].startswith("stem_") else "main_grid"
        if f is None or f["family"] != fam or {k: v for k, v in unit_spec(f).items() if k != "family"} != \
                {k: v for k, v in unit_spec(u).items() if k != "family"}:
            bad.append((u["unit_id"], "no frozen unit with the anchor's own spec"))
            continue
        wanted[u["unit_id"]] = f
    frozen = {}
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(args.frozen_archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and len(m.name.split("/")) > 3
                       and m.name.split("/")[1] == "units" and m.name.split("/")[2] in wanted]
            tar.extractall(tmp, members=members, filter="data")
        for uid, f in wanted.items():
            udir = Path(tmp) / receipt["run_id"] / "units" / uid
            reason = validate_result(f, udir, frozen_ident)
            if reason is not None:
                bad.append((f"frozen:{uid}", reason))
                continue
            try:
                m = unit_metrics(udir)
            except (ValueError, KeyError, IndexError, OSError) as exc:
                bad.append((f"frozen:{uid}", str(exc)))
                continue
            m["history_bytes"] = m["history_path"].read_bytes()
            m["history_rows"] = read_history(m["history_path"])
            frozen[uid] = (f, m)
    if bad:
        return fail(bad, f"{len(plan['units'])} planned units and their frozen references")

    out = Path(args.out)
    anchors = []
    for (fam, cond, model, seed), (u, m) in sorted(metrics.items()):
        if fam != FAM_ANCHOR:
            continue
        f, fm = frozen[u["unit_id"]]
        rows = read_history(m["history_path"])
        cols_equal = len(rows) == len(fm["history_rows"]) and all(
            x["epoch"] == y["epoch"] and all(x[k] == y[k] for k in METRIC_COLUMNS) for x, y in zip(rows, fm["history_rows"]))
        anchors.append({"anchor_unit": u["unit_id"], "frozen_family": f["family"], "dataset": u["dataset"], "model": model, "seed": seed,
                        "bytes_equal": m["history_path"].read_bytes() == fm["history_bytes"], "metric_columns_equal": cols_equal,
                        "params_equal": m["effective_params"] == fm["effective_params"], "data_equal": m["data_hashes"] == fm["data_hashes"],
                        "anchor_history_sha256": m["history_sha256"], "frozen_history_sha256": fm["history_sha256"]})
    anchors_pass = len(anchors) == len(ANCHORS) and all(a["bytes_equal"] and a["params_equal"] and a["data_equal"] for a in anchors)

    table = [{"source": "run", "family": u["family"], "unit_id": u["unit_id"], "condition": u["condition"], "dataset": u["dataset"],
              "model": u["model"], "seed": u["seed"], **{k: v for k, v in m.items() if k not in ("history_path", "data_hashes")}}
             for _key, (u, m) in sorted(metrics.items(), key=lambda kv: str(kv[0]))]
    data_hashes = {}
    for (_f, _c, _m, _s), (u, m) in sorted(metrics.items(), key=lambda kv: str(kv[0])):
        data_hashes.setdefault(u["dataset"], json.loads(m["data_hashes"]))

    summary = []
    for k, (fam, cond, model) in enumerate(CELLS):
        ms = [metrics[(fam, cond, model, s)][1] for s in SEEDS]
        c = [m["test_correct"] for m in ms]
        lo, hi = mean_ci(c, CELL_SEED0 + k)
        summary.append({"family": fam, "condition": cond, "dataset": DATASET_OF[cond], "model": model, "n_seeds": len(c),
                        "acc_mean_pp": 100.0 * statistics.mean(c) / N_TEST, "acc_sd_pp": 100.0 * statistics.stdev(c) / N_TEST,
                        "acc_min_pp": 100.0 * min(c) / N_TEST, "acc_max_pp": 100.0 * max(c) / N_TEST,
                        "acc_ci95_low_pp": 100.0 * lo / N_TEST, "acc_ci95_high_pp": 100.0 * hi / N_TEST,
                        "bootstrap_seed": CELL_SEED0 + k, "bootstrap_resamples": BOOT_R, "effective_params": ms[0]["effective_params"],
                        "train_acc_last_epoch_mean": statistics.mean(m["train_acc_last_epoch"] for m in ms),
                        "selected_epoch_mean": statistics.mean(m["best_val_epoch"] for m in ms),
                        "selected_last_epoch_count": sum(1 for m in ms if m["selected_last_epoch"]),
                        "unit_seconds_mean": statistics.mean(float(m["unit_seconds"] or 0.0) for m in ms)})

    results = []
    for k, (cid, group, fam, cond, a, b, decision) in enumerate(CONTRASTS):
        d = [metrics[(fam, cond, a, s)][1]["test_correct"] - metrics[(fam, cond, b, s)][1]["test_correct"] for s in SEEDS]
        st = int_stats(d, CONTRAST_SEED0 + k, N_TEST)
        results.append({"id": cid, "holm_group": group, "budget_epochs": EPOCHS_OF_FAMILY.get(fam, BASE["epochs"]), "condition": cond,
                        "A": a, "B": b, "decision_kind": decision, "direction": DIRECTION, **st,
                        "differences_count": ";".join(str(x) for x in d)})
    for group in HOLM_GROUPS:
        rows = [r for r in results if r["holm_group"] == group]
        for r, p in zip(rows, holm([x["p_two_sided"] for x in rows])):
            r["p_holm"] = p
            r["outcome"] = label(p, r)
    decisions = [{"criterion_id": r["id"], "holm_group": r["holm_group"], "budget_epochs": r["budget_epochs"], "n": r["n"],
                  "mean_diff_pp": r["mean_diff_pp"], "ci_low_pp": r["ci95_low_pp"], "ci_high_pp": r["ci95_high_pp"], "holm_p": r["p_holm"],
                  "d_z": r["d_z"], "rank_biserial": r["rank_biserial"], "wilcoxon_method": r["wilcoxon_method"], "threshold": THRESHOLD,
                  "direction": DIRECTION, "outcome": r["outcome"],
                  "discriminator": r["decision_kind"] in ("transfer", "increment") and r["holm_group"] in PRIMARY_GROUPS}
                 for r in results]
    by_id = {r["id"]: r for r in results}
    comp = []
    for d1_group, suffix in (("D1", ""), ("D1_e100", "_e100")):
        nb = [r for r in results if r["decision_kind"] == "transfer" and r["holm_group"] == d1_group]
        sup = sum(1 for r in nb if r["outcome"] == "supported")
        comp.append({"rule_id": f"D1_transfer{suffix}", "inputs": ";".join(f"{r['id']}={r['outcome']}" for r in nb), "supported": sup,
                     "reverse": sum(1 for r in nb if r["outcome"] == "reverse"),
                     "outcome": "transfer_observed" if sup == 2 else "dataset_specific" if sup == 1 else "not_observed",
                     "reverse_named": ";".join(r["condition"] for r in nb if r["outcome"] == "reverse"),
                     "rule": "the F9 rule: observed if both Naive-Branch-head contrasts are supported, dataset-specific if exactly one, "
                             "not observed otherwise; every reverse is named"
                             + (" (100-epoch budget arm, A17)" if suffix else " (primary, 30-epoch protocol)"), "boundary": BOUNDARY})
    for cond, suffix in ((D2_CONDITION, ""), (D2_CONDITION_E100, "_e100")):
        d2 = by_id[f"D2{suffix}:{cond}:dann_lrf-naive_branch"]
        comp.append({"rule_id": f"D2_increment{suffix}", "inputs": f"{d2['id']}={d2['outcome']}", "outcome": d2["outcome"],
                     "rule": "the A8 label of DANN-LRF minus Naive-Branch on CIFAR-100, reported as it is"
                             + (" (100-epoch budget arm, A17)" if suffix else " (primary, 30-epoch protocol)"), "boundary": BOUNDARY})
    pairs = [(by_id[primary_of(r["id"])], r) for r in results if r["holm_group"] not in PRIMARY_GROUPS]
    differing = [b["id"] for a, b in pairs if a["outcome"] != b["outcome"]]
    comp.append({"rule_id": "budget_agreement", "inputs": ";".join(f"{a['id']}={a['outcome']}|{b['outcome']}" for a, b in pairs),
                 "outcome": "budget_dependent" if differing else "consistent", "differing": ";".join(differing),
                 "rule": "DEC-NEURO-020: the primary label is the 30-epoch one; where the 100-epoch label of the same contrast differs, the "
                         "statement is worded as budget-dependent and both labels are reported", "boundary": BOUNDARY})

    write_csv(out / "unit_table.csv", table)
    write_csv(out / "summary_by_cell.csv", summary)
    write_csv(out / "contrasts.csv", results)
    write_csv(out / "decision_inputs.csv", decisions)
    write_csv(out / "composite_decisions.csv", comp)
    write_csv(out / "anchors.csv", anchors)
    ver = versions()
    run = {"script": "r1/aggregate_tierd.py", "script_sha256": sha_file(Path(__file__)),
           "statistics_source": "r1/aggregate_r1.py int_stats, holm, label", "statistics_source_sha256": sha_file(ROOT / "r1" / "aggregate_r1.py"),
           "plan_sha256": sha_file(Path(args.plan)), "identity_sha256": sha_file(Path(args.identity)),
           "frozen_plan_sha256": sha_file(Path(args.frozen_plan)), "frozen_archive_sha256": archive_sha,
           "frozen_identity_sha256": sha_file(Path(args.frozen_identity)), "units": len(metrics), "data_hashes": data_hashes,
           "anchors_pass": anchors_pass,
           "lineage": ("the run reproduces its nine frozen anchor units byte for byte" if anchors_pass else
                       "DETERMINISM FINDING: an anchor differs from its frozen unit; the within-run statistics stand, but no D1 or D2 "
                       "number is placed next to a frozen-run number as same-lineage evidence"),
           "contrasts": len(results), "contrast_bootstrap_seeds": {c[0]: CONTRAST_SEED0 + k for k, c in enumerate(CONTRASTS)},
           "cell_bootstrap_seeds": {f"{c[1]}:{c[2]}": CELL_SEED0 + k for k, c in enumerate(CELLS)},
           "composite": {r["rule_id"]: r["outcome"] for r in comp}, "boundary": BOUNDARY,
           "budget_epochs": {"primary": BASE["epochs"], "sensitivity_arm_A17": BUDGET_EPOCHS},
           "versions": ver, "versions_match_pinned": ver == PINNED}
    out.mkdir(parents=True, exist_ok=True)
    (out / "ANALYSIS_RUN.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8", newline="\n")
    cmp = run["composite"]
    print(f"units {len(metrics)}; anchors_pass {anchors_pass}; contrasts {len(results)}; D1 {cmp['D1_transfer']} "
          f"(100 epochs: {cmp['D1_transfer_e100']}); D2 increment {cmp['D2_increment']} (100 epochs: {cmp['D2_increment_e100']}); "
          f"budget agreement {cmp['budget_agreement']}; versions_match_pinned {run['versions_match_pinned']} -> {out}")
    return 0 if anchors_pass else 3


# ---------------------------------------------------------------- pilot (reference adequacy; pilot_only)
def pilot(args) -> int:
    try:
        plan = load_plan(Path(args.plan))
    except SystemExit as exc:
        return fail([("plan", str(exc))], "the pilot plan")
    problems = plan_problems(plan["units"], "pilot")
    if problems:
        return fail([("plan", p) for p in problems], "the pilot plan")
    bad = []
    metrics = collect(plan["units"], Path(args.root), read_json(args.identity) if args.identity else None, bad)
    if not bad:
        bad += data_problems(metrics)
    if bad:
        return fail(bad, "the pilot units")
    rows = []
    for (fam, cond, model, seed), (u, m) in sorted(metrics.items(), key=lambda kv: str(kv[0])):
        main_model = "tf_stem_dann_lrf" if model in TF_HEADS else "dann_lrf"
        main = metrics[(fam, cond, main_model, seed)][1]
        role = "main" if model == main_model else "reference"
        rising = m["selected_last_epoch"] and m["train_acc_change_last5"] > 0
        below = m["train_acc_last_epoch"] < main["test_acc"]
        rows.append({"unit_id": u["unit_id"], "condition": cond, "model": model, "seed": seed, "role": role,
                     "train_acc_last_epoch": m["train_acc_last_epoch"], "train_acc_change_last5": m["train_acc_change_last5"],
                     "selected_epoch": m["best_val_epoch"], "epochs": m["epochs"], "test_acc": m["test_acc"],
                     "main_model": main_model, "main_test_acc": main["test_acc"], "gap_to_main_pp": 100.0 * (main["test_acc"] - m["test_acc"]),
                     "flag_selected_last_still_rising": rising, "flag_train_fit_below_main_test": below,
                     "adequacy_flag": role == "reference" and (rising or below), "label": "pilot_only; never evidence"})
    out = Path(args.out)
    write_csv(out / "pilot_adequacy.csv", rows)
    flagged = [r["unit_id"] for r in rows if r["adequacy_flag"]]
    doc = {"script": "r1/aggregate_tierd.py pilot", "script_sha256": sha_file(Path(__file__)), "plan_sha256": sha_file(Path(args.plan)),
           "units": len(rows), "flagged_references": flagged, "label": "pilot_only; never evidence",
           "rule": "STUDY_DESIGN 'Baselines And Comparators': a reference selected at the last epoch with a still-rising curve, or "
                   "whose training fit is below the main model's test score, has not fitted; the decision is recorded before the lock"}
    (out / "PILOT_RUN.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"pilot units {len(rows)}; flagged references {flagged or 'none'} -> {out}")
    return 0


# ---------------------------------------------------------------- selftest (synthetic data only)
SELF_IDENTITY = {"freeze_id": "SELFTEST", "run_id": "selftest_tierd", "plan_sha256": "0" * 64}
SELF_FROZEN_IDENTITY = {"freeze_id": "SELFTEST-FROZEN", "run_id": "selftest_frozen", "plan_sha256": "1" * 64}
HEADER = ["dataset", "model_name", "seed", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]


def _unit(family, condition, model, seed, extra=None) -> dict:
    u = {"unit_id": f"{condition}__{model}__s{seed:02d}", "family": family, "condition": condition, "dataset": DATASET_OF[condition],
         "subset_fraction": 1.0, "model": model, "seed": seed, "extra": dict(extra or {})}
    u.update({k: v for k, v in BASE.items() if k != "subset_fraction"})
    if family in EPOCHS_OF_FAMILY:
        u["epochs"] = EPOCHS_OF_FAMILY[family]
    return u


def _plan_units(kind: str) -> list:
    if kind == "pilot":
        return [_unit(FAM_PILOT, c, m, PILOT_SEED, {"r1_model": m}) for c in D1_CONDITIONS for m in TF_HEADS] + \
               [_unit(FAM_PILOT, D2_CONDITION, m, PILOT_SEED) for m in D2_MODELS]
    units = [_unit(FAM_ANCHOR, c, m, 0, {"r1_model": m} if m.startswith("stem_") else {}) for c, m in ANCHORS]
    for s in SEEDS:
        units += [_unit(FAM_D1, c, m, s, {"r1_model": m}) for c in D1_CONDITIONS for m in TF_HEADS]
        units += [_unit(FAM_D2, D2_CONDITION, m, s) for m in D2_MODELS]
    for s in SEEDS:  # A17: the 100-epoch arm after the primary units, in the order of r1_plan.tierd_run
        units += [_unit(FAM_D1_E100, c, m, s, {"r1_model": m}) for c in D1_CONDITIONS_E100 for m in TF_HEADS]
        units += [_unit(FAM_D2_E100, D2_CONDITION_E100, m, s) for m in D2_MODELS]
    return units


def _write_unit(root: Path, u: dict, identity: dict, count: int, params: int, best_epoch: int = 20, train_last: float = 0.9,
                rising: bool = True, test_acc_override=None, data_tag: str = "") -> None:
    udir = root / "units" / u["unit_id"]
    (udir / "attempt_01").mkdir(parents=True, exist_ok=True)
    epochs = u["epochs"]
    rows = []
    for e in range(1, epochs + 1):
        test = count if e == best_epoch else max(0, count - 3 * abs(e - best_epoch) - 1)
        acc = test / N_TEST if (e != best_epoch or test_acc_override is None) else test_acc_override
        rows.append([u["dataset"], u["model"], u["seed"], e, 2.0 - 0.02 * e, (train_last - 0.005 * (epochs - e)) if rising else train_last,
                     1.0 + 0.01 * abs(e - best_epoch), 0.5, 1.5 + 0.001 * e, acc])
    hist = udir / "attempt_01" / "history.csv"
    with open(hist, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    best = rows[best_epoch - 1]
    summary = {"trainable_params": params, "best_val_epoch": best_epoch, "best_val_loss": best[6], "test_acc_at_best_val": best[9],
               "test_loss_at_best_val": best[8], "best_test_acc": max(r[9] for r in rows), "min_test_loss": min(r[8] for r in rows),
               "final_test_acc": rows[-1][9], "final_test_loss": rows[-1][8]}
    r1_kind = bool((u.get("extra") or {}).get("r1_model"))
    result = {"schema_version": 1, "status": "completed", "unit_id": u["unit_id"], "unit_spec": unit_spec(u),
              "unit_spec_sha256": unit_spec_sha256(u), "attempt": 1, "history_file": "attempt_01/history.csv",
              "history_sha256": sha_file(hist), "summary": summary,
              "model_info": {"dense_trainable_params": params, "effective_trainable_params": params} if r1_kind else None,
              "identity": identity, "timing_seconds": {"unit_total": 1.5},
              "data_hashes": {k: f"SYN-{u['dataset']}-{k}{data_tag}" for k in ("train_x", "train_y", "test_x", "test_y")}}
    (udir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _counts() -> dict:
    """Known effects: D1 FashionMNIST supported (exact) and an alternating MLP contrast; D1 CIFAR-10 reverse and an all-zero
    MLP contrast; D2 supported with ties and zeros (approximation), MLP-Param better (reverse), DANN-RANDOM alternating.
    100-epoch arm (A17): every count 1,000 higher with the same differences, except the CIFAR-10 transfer contrast, which
    alternates (not supported), so exactly one contrast differs between the two budgets."""
    tz = [0, 2, 2, 3, 3, 3, -1, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 12, 0]
    c = {}
    for s in SEEDS:
        alt = (s + 1) * (1 if s % 2 == 0 else -1)
        c[("fashion_full", "tf_stem_naive_branch", s)] = 8000 + 13 * s
        c[("fashion_full", "tf_stem_dann_lrf", s)] = 8000 + 13 * s + 31 + s
        c[("fashion_full", "tf_stem_mlp", s)] = 8000 + 13 * s + 31 + s - alt
        c[("cifar_full", "tf_stem_dann_lrf", s)] = 5000 + 11 * s
        c[("cifar_full", "tf_stem_naive_branch", s)] = 5000 + 11 * s + 40 + s
        c[("cifar_full", "tf_stem_mlp", s)] = 5000 + 11 * s
        c[(D2_CONDITION, "naive_branch", s)] = 2000 + 5 * s
        c[(D2_CONDITION, "dann_lrf", s)] = 2000 + 5 * s + tz[s]
        c[(D2_CONDITION, "mlp_param", s)] = 2000 + 5 * s + tz[s] + 60 + 2 * s
        c[(D2_CONDITION, "dann_random", s)] = 2000 + 5 * s + tz[s] + alt
        c[(D2_CONDITION, "vann_same", s)] = 2500 + s
    for (cond, m, s), v in list(c.items()):
        c[(cond + E100, m, s)] = v + 1000
    for s in SEEDS:
        alt = (s + 1) * (1 if s % 2 == 0 else -1)
        c[("cifar_full" + E100, "tf_stem_naive_branch", s)] = c[("cifar_full" + E100, "tf_stem_dann_lrf", s)] + alt
    return c


ANCHOR_PARAMS = {"dann_lrf": 10634, "naive_branch": 10634, "mlp_param": 10507, "dann_random": 10634, "vann_same": 1639050,
                 "stem_dann_lrf": 11882, "stem_naive_branch": 12026, "stem_mlp": 11862}  # synthetic values; only equality matters


def _build_synthetic(tmp: Path) -> dict:
    counts = _counts()
    units = _plan_units("arm")
    run = tmp / "run"
    for u in units:
        if u["family"] == FAM_ANCHOR:
            _write_unit(run, u, SELF_IDENTITY, 6000 + len(u["model"]), ANCHOR_PARAMS[u["model"]], best_epoch=25)
        else:
            _write_unit(run, u, SELF_IDENTITY, counts[(u["condition"], u["model"], u["seed"])], EXPECTED_PARAMS[(u["condition"], u["model"])],
                        best_epoch=20 + u["seed"] % 11)
    plan = {"schema_version": 1, "run_id": "selftest_tierd", "family": "tierd_run", "units": units}
    (tmp / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    (tmp / "identity.json").write_text(json.dumps(SELF_IDENTITY) + "\n", encoding="utf-8", newline="\n")
    # frozen side: the nine counterparts plus two unrelated units, under the frozen families and identity
    frozen_units = [dict(u, family="F9_convstem" if u["model"].startswith("stem_") else "main_grid") for u in units if u["family"] == FAM_ANCHOR]
    frozen_units += [_unit("main_grid", "fashion_full", "dann_lrf", 0), _unit("F9_convstem", "fashion_full", "stem_mlp", 0, {"r1_model": "stem_mlp"})]
    frun = tmp / "frozen_build" / "selftest_frozen"
    for u in frozen_units:
        _write_unit(frun, u, SELF_FROZEN_IDENTITY, 6000 + len(u["model"]), ANCHOR_PARAMS[u["model"]], best_epoch=25)
    (tmp / "frozen_plan.json").write_text(json.dumps({"schema_version": 1, "run_id": "selftest_frozen", "units": frozen_units}, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    (tmp / "frozen_identity.json").write_text(json.dumps(SELF_FROZEN_IDENTITY) + "\n", encoding="utf-8", newline="\n")
    _archive(tmp / "frozen_build", tmp / "frozen.tar.gz", tmp / "frozen_receipt.json")
    return {"tmp": tmp, "run": run, "counts": counts}


def _archive(build_dir: Path, archive: Path, receipt: Path, skip_unit=None) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted((build_dir / "selftest_frozen").rglob("*")):
            if p.is_file() and (skip_unit is None or skip_unit not in p.parts):
                tar.add(p, arcname=p.relative_to(build_dir).as_posix())
    receipt.write_text(json.dumps({"run_id": "selftest_frozen", "archive": {"sha256": sha_file(archive)}}) + "\n", encoding="utf-8", newline="\n")


def _ns(tmp: Path, out: Path, **over) -> argparse.Namespace:
    d = {"plan": tmp / "plan.json", "root": tmp / "run", "identity": tmp / "identity.json", "frozen_plan": tmp / "frozen_plan.json",
         "frozen_archive": tmp / "frozen.tar.gz", "frozen_receipt": tmp / "frozen_receipt.json",
         "frozen_identity": tmp / "frozen_identity.json", "out": out}
    d.update(over)
    return argparse.Namespace(**{k: str(v) for k, v in d.items()})


def _quiet(fn, ns):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(ns)
    return rc, buf.getvalue()


def _read_csv(path: Path) -> list:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _exact_p(d) -> float:
    """Two-sided exact signed-rank p by dynamic programming over subset sums (no zeros, no tied |d|)."""
    n = len(d)
    rank = {a: i + 1 for i, a in enumerate(sorted(abs(x) for x in d))}
    w = sum(rank[abs(x)] for x in d if x > 0)
    total = n * (n + 1) // 2
    ways = [1] + [0] * total
    for r in range(1, n + 1):
        for s in range(total, r - 1, -1):
            ways[s] += ways[s - r]
    return min(1.0, 2.0 * min(sum(ways[: w + 1]), sum(ways[w:])) / 2 ** n)


def _avg_ranks(nz):
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks, ties, i = [0.0] * len(nz), [], 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def _approx_p(d) -> float:
    """Normal approximation, zeros dropped, tie-corrected variance, no continuity correction."""
    nz = [x for x in d if x != 0]
    n = len(nz)
    ranks, ties = _avg_ranks(nz)
    w = sum(r for r, x in zip(ranks, nz) if x > 0)
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties) / 48
    return math.erfc(abs(w - n * (n + 1) / 4) / math.sqrt(var) / math.sqrt(2))


def _rank_biserial(d) -> float:
    nz = [x for x in d if x != 0]
    if not nz:
        return 0.0
    ranks, _ = _avg_ranks(nz)
    wp = sum(r for r, x in zip(ranks, nz) if x > 0)
    wn = sum(r for r, x in zip(ranks, nz) if x < 0)
    return (wp - wn) / (wp + wn)


def _holm(ps):
    idx, out, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = run
    return out


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def selftest(_args) -> int:
    import numpy as np

    fails = []

    def check(name: str, cond: bool) -> None:
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        env = _build_synthetic(tmp)
        counts = env["counts"]
        rc, log = _quiet(analyse, _ns(tmp, tmp / "out_ok"))
        check("analyse on a complete synthetic run exits 0", rc == 0)
        out = tmp / "out_ok"
        con = {r["id"]: r for r in _read_csv(out / "contrasts.csv")}
        expected = {"D1:fashion_full:tf_stem_dann_lrf-tf_stem_naive_branch": ("supported", "exact"),
                    "D1:fashion_full:tf_stem_dann_lrf-tf_stem_mlp": ("not_supported", "exact"),
                    "D1:cifar_full:tf_stem_dann_lrf-tf_stem_naive_branch": ("reverse", "exact"),
                    "D1:cifar_full:tf_stem_dann_lrf-tf_stem_mlp": ("no_difference", "all_zero"),
                    "D2:cifar100_full:dann_lrf-naive_branch": ("supported", "approx"),
                    "D2:cifar100_full:dann_lrf-mlp_param": ("reverse", "exact"),
                    "D2:cifar100_full:dann_lrf-dann_random": ("not_supported", "exact"),
                    "D1_e100:fashion_full_e100:tf_stem_dann_lrf-tf_stem_naive_branch": ("supported", "exact"),
                    "D1_e100:fashion_full_e100:tf_stem_dann_lrf-tf_stem_mlp": ("not_supported", "exact"),
                    "D1_e100:cifar_full_e100:tf_stem_dann_lrf-tf_stem_naive_branch": ("not_supported", "exact"),
                    "D1_e100:cifar_full_e100:tf_stem_dann_lrf-tf_stem_mlp": ("no_difference", "all_zero"),
                    "D2_e100:cifar100_full_e100:dann_lrf-naive_branch": ("supported", "approx"),
                    "D2_e100:cifar100_full_e100:dann_lrf-mlp_param": ("reverse", "exact"),
                    "D2_e100:cifar100_full_e100:dann_lrf-dann_random": ("not_supported", "exact")}
        check("fourteen pre-specified contrasts in the fixed order (seven primary, seven at 100 epochs)",
              list(con) == [c[0] for c in CONTRASTS] == list(expected))
        check("the 100-epoch contrasts carry budget 100 and the primary ones budget 30",
              all(int(con[c[0]]["budget_epochs"]) == (BUDGET_EPOCHS if c[1].endswith("_e100") else 30) for c in CONTRASTS))
        for cid, (lab, method) in expected.items():
            check(f"label {cid} = {lab} ({method})", con[cid]["outcome"] == lab and con[cid]["wilcoxon_method"] == method)
        # independent recomputation of every statistic
        raw_p, pairs = {}, {}
        for k, (cid, group, _fam, cond, a, b, _dec) in enumerate(CONTRASTS):
            d = [counts[(cond, a, s)] - counts[(cond, b, s)] for s in SEEDS]
            pairs[cid] = d
            row = con[cid]
            check(f"{cid}: differences written equal the synthetic differences", row["differences_count"] == ";".join(map(str, d)))
            if row["wilcoxon_method"] == "exact":
                raw_p[cid] = _exact_p(d)
            elif row["wilcoxon_method"] == "approx":
                raw_p[cid] = _approx_p(d)
            else:
                raw_p[cid] = 1.0
            check(f"{cid}: two-sided p equals the independent recomputation ({raw_p[cid]:.6g})", _close(float(row["p_two_sided"]), raw_p[cid]))
            arr = np.asarray(d, dtype=np.int64)
            means = arr[np.random.default_rng(CONTRAST_SEED0 + k).integers(0, 20, size=(10_000, 20))].mean(axis=1)
            check(f"{cid}: bootstrap CI with seed {CONTRAST_SEED0 + k}",
                  float(row["ci95_low_count"]) == float(np.quantile(means, 0.025)) and float(row["ci95_high_count"]) == float(np.quantile(means, 0.975))
                  and int(row["bootstrap_seed"]) == CONTRAST_SEED0 + k)
            sd = statistics.stdev(d)
            dz_ok = row["d_z"] == "undefined" if sd == 0 else _close(float(row["d_z"]), statistics.mean(d) / sd)
            check(f"{cid}: d_z and rank-biserial", dz_ok and _close(float(row["rank_biserial"]), _rank_biserial(d))
                  and _close(float(row["mean_diff_pp"]), 100.0 * statistics.mean(d) / N_TEST))
        for group in HOLM_GROUPS:
            ids = [c[0] for c in CONTRASTS if c[1] == group]
            adj = _holm([raw_p[i] for i in ids])
            check(f"Holm within {group} ({len(ids)} contrasts) equals the independent step-down",
                  all(_close(float(con[i]["p_holm"]), p) for i, p in zip(ids, adj)))
        comp = {r["rule_id"]: r for r in _read_csv(out / "composite_decisions.csv")}
        check("D1 composite: dataset_specific with the CIFAR-10 reverse named",
              comp["D1_transfer"]["outcome"] == "dataset_specific" and comp["D1_transfer"]["reverse_named"] == "cifar_full")
        check("D2 composite: the increment label is reported as it is", comp["D2_increment"]["outcome"] == "supported")
        check("D1 composite at 100 epochs: dataset_specific with no reverse named",
              comp["D1_transfer_e100"]["outcome"] == "dataset_specific" and comp["D1_transfer_e100"]["reverse_named"] == "")
        check("D2 composite at 100 epochs: supported", comp["D2_increment_e100"]["outcome"] == "supported")
        check("budget agreement: budget_dependent, and exactly the CIFAR-10 transfer contrast differs",
              comp["budget_agreement"]["outcome"] == "budget_dependent"
              and comp["budget_agreement"]["differing"] == "D1_e100:cifar_full_e100:tf_stem_dann_lrf-tf_stem_naive_branch")
        summ = _read_csv(out / "summary_by_cell.csv")
        check("twenty-two cells (eleven per budget) with means, parameter counts and seeds as pre-specified",
              len(summ) == 22 and all(_close(float(r["acc_mean_pp"]), 100.0 * statistics.mean(counts[(r["condition"], r["model"], s)] for s in SEEDS) / N_TEST)
                                      and int(r["effective_params"]) == EXPECTED_PARAMS[(r["condition"], r["model"])]
                                      and int(r["bootstrap_seed"]) == CELL_SEED0 + k for k, r in enumerate(summ)))
        check("selected-at-last-epoch count (best epoch 20 + seed mod 11: 1 of 20 at 30 epochs, 0 of 20 at 100 epochs)",
              all(int(r["selected_last_epoch_count"]) == (0 if r["condition"].endswith(E100) else 1) for r in summ))
        boot = [CONTRAST_SEED0 + k for k in range(len(CONTRASTS))] + [CELL_SEED0 + k for k in range(len(CELLS))]
        check("bootstrap generator seeds of the fourteen contrasts and twenty-two cells are distinct", len(set(boot)) == len(boot) == 36)
        anchors = _read_csv(out / "anchors.csv")
        run = read_json(out / "ANALYSIS_RUN.json")
        check("nine anchors byte-equal; anchors_pass true; 449 units",
              len(anchors) == 9 and all(a["bytes_equal"] == "True" for a in anchors) and run["anchors_pass"] is True and run["units"] == 449)
        dec = _read_csv(out / "decision_inputs.csv")
        check("decision inputs carry the direction and the threshold; only the primary transfer and increment discriminate",
              len(dec) == 14 and all(r["direction"] == DIRECTION and r["threshold"] == THRESHOLD for r in dec)
              and sorted(r["criterion_id"] for r in dec if r["discriminator"] == "True") ==
              sorted(["D1:fashion_full:tf_stem_dann_lrf-tf_stem_naive_branch", "D1:cifar_full:tf_stem_dann_lrf-tf_stem_naive_branch",
                      "D2:cifar100_full:dann_lrf-naive_branch"]))

        # fail-closed paths: each starts from a fresh copy of the valid synthetic run
        def variant(name: str):
            vt = tmp / name
            shutil.copytree(tmp / "run", vt / "run")
            for f in ("plan.json", "identity.json", "frozen_plan.json", "frozen.tar.gz", "frozen_receipt.json", "frozen_identity.json"):
                shutil.copy2(tmp / f, vt / f)
            shutil.copytree(tmp / "frozen_build", vt / "frozen_build")
            return vt

        def refuse(name: str, vt: Path, reason: str, **over) -> None:
            rc, log = _quiet(analyse, _ns(vt, vt / "out", **over))
            check(f"fail closed: {name} -> exit 2, nothing written, reason '{reason}' reported",
                  rc == 2 and not (vt / "out").exists() and reason in log)

        vt = variant("missing_unit")
        (vt / "run" / "units" / "cifar100_full__vann_same__s07" / "result.json").unlink()
        refuse("a planned unit without result.json", vt, "cifar100_full__vann_same__s07 no result.json")
        vt = variant("missing_in_plan")
        p = read_json(vt / "plan.json")
        p["units"] = [u for u in p["units"] if u["unit_id"] != "fashion_full__tf_stem_mlp__s03"]
        (vt / "plan.json").write_text(json.dumps(p), encoding="utf-8")
        refuse("a pre-specified unit missing from the plan", vt, "missing from the plan: ('MC003_D1_tfstem', 'fashion_full', 'tf_stem_mlp', 3)")
        vt = variant("duplicate_id")
        p = read_json(vt / "plan.json")
        p["units"].append(dict(p["units"][20]))
        (vt / "plan.json").write_text(json.dumps(p), encoding="utf-8")
        refuse("a duplicate unit id", vt, "duplicate unit_id in plan")
        vt = variant("duplicate_cell")
        p = read_json(vt / "plan.json")
        p["units"].append(dict(p["units"][20], unit_id=p["units"][20]["unit_id"] + "_again"))
        (vt / "plan.json").write_text(json.dumps(p), encoding="utf-8")
        refuse("a duplicate cell and seed under another unit id", vt, "duplicate cell and seed")
        vt = variant("changed_epochs")
        p = read_json(vt / "plan.json")
        p["units"][30]["epochs"] = 29
        (vt / "plan.json").write_text(json.dumps(p), encoding="utf-8")
        refuse("a changed hyper-parameter", vt, "epochs=29, pre-specified 30")
        vt = variant("changed_epochs_e100")
        p = read_json(vt / "plan.json")
        i = next(i for i, x in enumerate(p["units"]) if x["unit_id"] == "cifar100_full_e100__dann_lrf__s05")
        p["units"][i]["epochs"] = 30
        (vt / "plan.json").write_text(json.dumps(p), encoding="utf-8")
        refuse("a 100-epoch unit planned at 30 epochs", vt, "epochs=30, pre-specified 100")
        vt = variant("receipt")
        (vt / "frozen_receipt.json").write_text(json.dumps({"run_id": "selftest_frozen", "archive": {"sha256": "0" * 64}}), encoding="utf-8")
        refuse("a frozen archive that does not hash to its receipt", vt, "does not hash to its receipt")
        vt = variant("frozen_missing")
        _archive(vt / "frozen_build", vt / "frozen.tar.gz", vt / "frozen_receipt.json", skip_unit="cifar_full__stem_mlp__s00")
        refuse("a missing frozen anchor reference", vt, "frozen:cifar_full__stem_mlp__s00 no result.json")
        vt = variant("identity")
        (vt / "identity.json").write_text(json.dumps(dict(SELF_IDENTITY, run_id="another_run")), encoding="utf-8")
        refuse("an identity record that differs from the units'", vt, "identity mismatch")
        vt = variant("non_integer")
        u = next(x for x in _plan_units("arm") if x["unit_id"] == "cifar_full__tf_stem_dann_lrf__s04")
        shutil.rmtree(vt / "run" / "units" / u["unit_id"])
        _write_unit(vt / "run", u, SELF_IDENTITY, 5044, 14698, best_epoch=24, test_acc_override=0.50443)
        refuse("an accuracy that is not a whole number of test images", vt, "is not a whole number of the 10000 test images")
        vt = variant("params")
        u = next(x for x in _plan_units("arm") if x["unit_id"] == "fashion_full__tf_stem_mlp__s09")
        shutil.rmtree(vt / "run" / "units" / u["unit_id"])
        _write_unit(vt / "run", u, SELF_IDENTITY, counts[("fashion_full", "tf_stem_mlp", 9)], 13840, best_epoch=20 + 9)
        refuse("a parameter count that differs from the pre-specified one", vt, "13840 trainable parameters, pre-specified 13839")
        vt = variant("extra_folder")
        shutil.copytree(vt / "run" / "units" / "cifar100_full__dann_lrf__s00", vt / "run" / "units" / "cifar100_full__dann_lrf__s20")
        refuse("a unit folder the plan does not name", vt, "cifar100_full__dann_lrf__s20 unit folder not named by the plan")
        vt = variant("data_mismatch")
        u = next(x for x in _plan_units("arm") if x["unit_id"] == "cifar100_full__mlp_param__s12")
        shutil.rmtree(vt / "run" / "units" / u["unit_id"])
        _write_unit(vt / "run", u, SELF_IDENTITY, counts[(D2_CONDITION, "mlp_param", 12)], 22367, best_epoch=20 + 12 % 11, data_tag="-other")
        refuse("converted data tensors that differ between units of one dataset", vt, "cifar100 data tensors differ between units (2 hash sets)")
        vt = variant("anchor_mismatch")
        u = next(x for x in _plan_units("arm") if x["unit_id"] == "cifar_full__naive_branch__s00")
        shutil.rmtree(vt / "run" / "units" / u["unit_id"])
        _write_unit(vt / "run", u, SELF_IDENTITY, 6000 + len(u["model"]), ANCHOR_PARAMS[u["model"]], best_epoch=25, train_last=0.91)
        rc, _ = _quiet(analyse, _ns(vt, vt / "out"))
        run_m = read_json(vt / "out" / "ANALYSIS_RUN.json") if (vt / "out" / "ANALYSIS_RUN.json").is_file() else {}
        check("anchor mismatch -> exit 3, outputs written, anchors_pass false, statistics unchanged",
              rc == 3 and run_m.get("anchors_pass") is False and (vt / "out" / "contrasts.csv").read_bytes() == (out / "contrasts.csv").read_bytes())

        # pilot: two flagged references and the main models unflagged
        pt = tmp / "pilot"
        punits = _plan_units("pilot")
        for u in punits:
            main = u["model"] in ("tf_stem_dann_lrf", "dann_lrf")
            best, train_last, rising = 22, 0.95, False
            if u["model"] == "tf_stem_naive_branch" and u["condition"] == "cifar_full":
                best, rising = 30, True  # selected at the last epoch with a rising training curve
            if u["model"] == "mlp_param":
                train_last = 0.10  # training fit below the main model's test accuracy (0.25)
            _write_unit(pt / "run", u, SELF_IDENTITY, 2500 if main else 2400, EXPECTED_PARAMS[(u["condition"], u["model"])],
                        best_epoch=best, train_last=train_last, rising=rising)
        (pt / "plan.json").write_text(json.dumps({"schema_version": 1, "run_id": "selftest_pilot", "units": punits}), encoding="utf-8")
        rc, _ = _quiet(pilot, argparse.Namespace(plan=str(pt / "plan.json"), root=str(pt / "run"), identity=str(tmp / "identity.json"), out=str(pt / "out")))
        prow = {r["unit_id"]: r for r in _read_csv(pt / "out" / "pilot_adequacy.csv")} if rc == 0 else {}
        flagged = sorted(k for k, r in prow.items() if r["adequacy_flag"] == "True")
        check("pilot: exit 0, eleven rows, flags on exactly the two constructed references",
              rc == 0 and len(prow) == 11 and flagged == ["cifar100_full__mlp_param__s100", "cifar_full__tf_stem_naive_branch__s100"])
        bad_plan = [dict(u, seed=5, unit_id=u["unit_id"].replace("s100", "s05")) for u in punits]
        (pt / "plan_bad.json").write_text(json.dumps({"schema_version": 1, "run_id": "selftest_pilot", "units": bad_plan}), encoding="utf-8")
        rc, _ = _quiet(pilot, argparse.Namespace(plan=str(pt / "plan_bad.json"), root=str(pt / "run"), identity=None, out=str(pt / "out_bad")))
        check("pilot: a plan with an evidence seed is refused (exit 2)", rc == 2 and not (pt / "out_bad").exists())
    ver = versions()
    print(f"versions {ver}; pinned {PINNED}; match {ver == PINNED}")
    print("FAILED:", fails if fails else "none")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse")
    for name in ("plan", "root", "identity", "frozen-plan", "frozen-archive", "frozen-receipt", "frozen-identity", "out"):
        a.add_argument(f"--{name}", required=True)
    p = sub.add_parser("pilot")
    for name in ("plan", "root", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--identity", default=None)
    sub.add_parser("selftest")
    args = ap.parse_args()
    return {"analyse": analyse, "pilot": pilot, "selftest": selftest}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
