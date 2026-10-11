"""Pre-specified analysis of the F1 convergence arm (MC-NEURO-R1-004; protocol Section 11, amendment A18; DEC-NEURO-021); no
model code. Written before any unit of the arm existed.

    python -B r1/aggregate_f1conv.py analyse --plan <plan_f1conv_run.json> --root <extracted run folder> --identity <IDENTITY.json>
        --frozen-plan <plan_frozen_all.json> --frozen-archive <frozen outputs.tar.gz> --frozen-receipt <frozen receipt.json>
        --frozen-identity <frozen IDENTITY.json> --frozen-contrasts <frozen processed_outputs/contrasts.csv> --out <dir>
    python -B r1/aggregate_f1conv.py selftest

analyse fails closed (exit 2, nothing written) when the plan is not the pre-specified 240-unit set (a missing, extra or duplicate
cell or seed; a unit that differs from its frozen 30-epoch counterpart in anything but the unit id, the condition, the family and
the number of epochs), when the run folder holds a unit the plan does not name, when any planned unit is missing or invalid
(r1_runner.validate_result with the identity record), when an accuracy is not a whole number of the 10,000 test images, when a
unit's trainable-parameter count or converted-data hashes differ from those of its frozen counterpart, when the frozen archive
does not hash to its receipt, when a frozen counterpart is missing or invalid, or when the frozen contrasts file lacks a label of a
contrast. Prefix gate: epochs 1-30 of every 100-epoch history must equal the frozen 30-epoch history row for row in every column;
a differing prefix is a determinism finding, the outputs are written and the exit code is 3.

Primary metric (A9): the integer number of correct test images at the first epoch of minimum validation loss over the 100 epochs.
Statistics: the frozen functions of r1/aggregate_r1.py (int_stats, holm, label). Contrasts per dataset in the order of the frozen
manifest: DANN-LRF minus MLP-Param, minus Naive-Branch, minus DANN-RANDOM; Holm within each dataset (the frozen F1 family
structure); bootstrap seed CONTRAST_SEED0 + k; labels by A8. Composites: the A8 two-of-three summary at 100 epochs (reported beside
the frozen verdict, never replacing it); the budget agreement of each contrast with its frozen 30-epoch label. Descriptive: the
per-seed budget change of each paired difference (100-epoch difference minus the frozen 30-epoch difference; mean with a bootstrap
CI, seed CHANGE_SEED0 + k; no test, no label) and, per cell, the accuracy gain over the frozen cell and the fit diagnostics.
Outputs: unit_table.csv, summary_by_cell.csv, contrasts.csv, decision_inputs.csv, budget_agreement.csv, budget_change.csv,
prefix_gate.csv, composite_decisions.csv, ANALYSIS_RUN.json.

selftest: a synthetic frozen archive and a synthetic arm; checks the outputs against an independent computation and every
fail-closed path. Exit 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
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

from r1.aggregate_r1 import BOOT_R, holm, int_stats, label, read_history, sha_file, write_csv  # noqa: E402
from r1.r1_runner import load_plan, unit_spec, unit_spec_sha256, validate_result  # noqa: E402

N_TEST = 10_000
SEEDS = list(range(20))
FAM = "MC004_F1_e100"
FROZEN_FAM = "main_grid"
EPOCHS = 100
PREFIX = 30
BASE_OF = {"fashion_full_e100": "fashion_full", "kmnist_full_e100": "kmnist_full", "cifar_full_e100": "cifar_full"}
DATASET_OF = {"fashion_full_e100": "fashionmnist", "kmnist_full_e100": "kmnist", "cifar_full_e100": "cifar10"}
CONDITIONS = tuple(BASE_OF)
MODELS = ("dann_lrf", "naive_branch", "mlp_param", "dann_random")
BASE = {"epochs": EPOCHS, "batch_size": 256, "lr": 0.001, "val_fraction": 0.1, "soma_units": 128, "branches_per_soma": 4,
        "sample_size": 16, "patch_h": 4, "patch_w": 4, "subset_fraction": 1.0}
# the frozen F1 full-data contrasts of these four models, in the order of R1_ANALYSIS_MANIFEST.json
CONTRASTS = [(f"F1e100:{c}:dann_lrf-{b}", f"F1e100:{c}", c, "dann_lrf", b, "rq2" if b == "naive_branch" else "label",
              f"F1:{BASE_OF[c]}:dann_lrf-{b}") for c in CONDITIONS for b in ("mlp_param", "naive_branch", "dann_random")]
CELLS = [(c, m) for c in CONDITIONS for m in MODELS]
CONTRAST_SEED0 = 20261005001
CELL_SEED0 = 20261005101
CHANGE_SEED0 = 20261005201
DIRECTION = "positive = A better than B (more test images correct)"
THRESHOLD = "holm_p < 0.05 and ci95_low > 0 (reverse: holm_p < 0.05 and ci95_high < 0)"
BOUNDARY = ("100-epoch convergence arm of F1 (MC-NEURO-R1-004, DEC-NEURO-021), reported whatever the outcome; the frozen 30-epoch "
            "labels of NEURO-R1-FREEZE-001 stay primary; a differing 100-epoch label is worded as budget-dependent with both labels; "
            "the arm changes no frozen verdict")
PINNED = {"python": "3.12.12", "numpy": "2.3.5", "scipy": "1.18.0"}
IGNORED_FOR_COUNTERPART = ("unit_id", "family", "condition", "epochs")


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_uid(u: dict) -> str:
    return f"{BASE_OF.get(u.get('condition'), '?')}__{u.get('model')}__s{int(u.get('seed', -1)):02d}"


def expected_keys() -> set:
    return {(FAM, c, m, s) for c in CONDITIONS for m in MODELS for s in SEEDS}


def plan_problems(units, frozen_by_id: dict) -> list:
    """Every deviation of the plan from the pre-specified unit set (A18)."""
    problems, seen = [], set()
    for u in units:
        key, uid = (u.get("family"), u.get("condition"), u.get("model"), u.get("seed")), u.get("unit_id")
        if key in seen:
            problems.append(f"duplicate cell and seed {key}")
        seen.add(key)
        if not isinstance(u.get("seed"), int) or uid != f"{u.get('condition')}__{u.get('model')}__s{u['seed']:02d}":
            problems.append(f"{uid}: unit id does not follow condition__model__sNN")
            continue
        if u.get("dataset") != DATASET_OF.get(u.get("condition")):
            problems.append(f"{uid}: dataset {u.get('dataset')!r} does not belong to condition {u.get('condition')!r}")
        for k, v in BASE.items():
            if u.get(k) != v:
                problems.append(f"{uid}: {k}={u.get(k)!r}, pre-specified {v!r}")
        if (u.get("extra") or {}) != {}:
            problems.append(f"{uid}: extra {u.get('extra')!r} is not empty")
        f = frozen_by_id.get(frozen_uid(u))
        if f is None or f.get("family") != FROZEN_FAM or int(f.get("epochs", -1)) != PREFIX:
            problems.append(f"{uid}: no frozen 30-epoch main-grid counterpart {frozen_uid(u)}")
            continue
        a = {k: v for k, v in unit_spec(u).items() if k not in IGNORED_FOR_COUNTERPART}
        b = {k: v for k, v in unit_spec(f).items() if k not in IGNORED_FOR_COUNTERPART}
        if a != b:
            problems.append(f"{uid}: differs from its frozen counterpart beyond unit id, condition, family and epochs")
    want = expected_keys()
    problems += [f"missing from the plan: {k}" for k in sorted(want - seen, key=str)]
    problems += [f"not pre-specified: {k}" for k in sorted(seen - want, key=str)]
    return problems


def unit_metrics(unit_dir: Path) -> dict:
    r = read_json(unit_dir / "result.json")
    hpath = unit_dir / r["history_file"]
    h = read_history(hpath)
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
            "history_rows": h, "history_bytes": hpath.read_bytes(), "history_sha256": r["history_sha256"]}


def mean_ci(values, seed: int):
    """95 % percentile bootstrap of the mean (the int_stats resampling rule, R = 10,000)."""
    import numpy as np

    v = np.asarray([int(x) for x in values], dtype=np.int64)
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
    frozen_by_id = {u["unit_id"]: u for u in frozen_plan["units"]}
    problems = plan_problems(plan["units"], frozen_by_id)
    if problems:
        return fail([("plan", p) for p in problems], "the plan")
    identity, frozen_ident, receipt = read_json(args.identity), read_json(args.frozen_identity), read_json(args.frozen_receipt)
    archive_sha = sha_file(Path(args.frozen_archive))
    if archive_sha != str(receipt["archive"]["sha256"]).upper():
        return fail([("frozen archive", "does not hash to its receipt")], "the frozen archive")
    with open(args.frozen_contrasts, encoding="utf-8", newline="") as f:
        frozen_rows = {r["id"]: r for r in csv.DictReader(f)}
    bad = [(c[6], "frozen contrast label missing") for c in CONTRASTS if c[6] not in frozen_rows or not frozen_rows[c[6]].get("outcome")]

    # frozen counterparts, read from the frozen archive
    wanted = {frozen_uid(u): frozen_by_id[frozen_uid(u)] for u in plan["units"]}
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
                frozen[uid] = unit_metrics(udir)
            except (ValueError, KeyError, IndexError, OSError) as exc:
                bad.append((f"frozen:{uid}", str(exc)))

    # the arm's units
    metrics = {}
    root = Path(args.root)
    for u in plan["units"]:
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
        fm = frozen.get(frozen_uid(u))
        if fm is not None:
            if m["effective_params"] != fm["effective_params"]:
                bad.append((u["unit_id"], f"{m['effective_params']} trainable parameters, frozen counterpart {fm['effective_params']}"))
                continue
            if m["data_hashes"] != fm["data_hashes"] or "null" in m["data_hashes"]:
                bad.append((u["unit_id"], "converted data tensors differ from the frozen counterpart's (or are not recorded)"))
                continue
        metrics[(u["condition"], u["model"], int(u["seed"]))] = (u, m)
    planned = {u["unit_id"] for u in plan["units"]}
    if (root / "units").is_dir():
        bad += [(p.name, "unit folder not named by the plan") for p in sorted((root / "units").iterdir()) if p.is_dir() and p.name not in planned]
    if bad:
        return fail(bad, f"{len(plan['units'])} planned units and their frozen counterparts")

    out = Path(args.out)
    # prefix gate (determinism and lineage): epochs 1-30 equal the frozen history row for row in every column
    gate = []
    for (cond, model, seed), (u, m) in sorted(metrics.items()):
        fm = frozen[frozen_uid(u)]
        rows, frows = m["history_rows"], fm["history_rows"]
        rows_equal = len(frows) == PREFIX and len(rows) == EPOCHS and all(rows[i] == frows[i] for i in range(PREFIX))
        first = m["history_bytes"].split(b"\n")[: PREFIX + 1]
        ffirst = fm["history_bytes"].split(b"\n")[: PREFIX + 1]
        gate.append({"unit_id": u["unit_id"], "frozen_unit_id": frozen_uid(u), "condition": cond, "model": model, "seed": seed,
                     "prefix_rows_equal": rows_equal, "prefix_bytes_equal": first == ffirst,
                     "history_sha256": m["history_sha256"], "frozen_history_sha256": fm["history_sha256"]})
    gate_pass = len(gate) == len(expected_keys()) and all(g["prefix_rows_equal"] for g in gate)

    table = [{"unit_id": u["unit_id"], "family": u["family"], "condition": u["condition"], "dataset": u["dataset"], "model": u["model"],
              "seed": u["seed"], **{k: v for k, v in m.items() if k not in ("history_rows", "history_bytes", "data_hashes")},
              "frozen_test_correct": frozen[frozen_uid(u)]["test_correct"], "frozen_best_val_epoch": frozen[frozen_uid(u)]["best_val_epoch"]}
             for _k, (u, m) in sorted(metrics.items())]
    data_hashes = {}
    for (_c, _m, _s), (u, m) in sorted(metrics.items()):
        data_hashes.setdefault(u["dataset"], json.loads(m["data_hashes"]))

    summary = []
    for k, (cond, model) in enumerate(CELLS):
        pairs = [metrics[(cond, model, s)] for s in SEEDS]
        ms = [m for _u, m in pairs]
        c = [m["test_correct"] for m in ms]
        fc = [frozen[frozen_uid(u)]["test_correct"] for u, _m in pairs]
        lo, hi = mean_ci(c, CELL_SEED0 + k)
        summary.append({"condition": cond, "dataset": DATASET_OF[cond], "model": model, "n_seeds": len(c),
                        "acc_mean_pp": 100.0 * statistics.mean(c) / N_TEST, "acc_sd_pp": 100.0 * statistics.stdev(c) / N_TEST,
                        "acc_min_pp": 100.0 * min(c) / N_TEST, "acc_max_pp": 100.0 * max(c) / N_TEST,
                        "acc_ci95_low_pp": 100.0 * lo / N_TEST, "acc_ci95_high_pp": 100.0 * hi / N_TEST,
                        "bootstrap_seed": CELL_SEED0 + k, "bootstrap_resamples": BOOT_R, "effective_params": ms[0]["effective_params"],
                        "frozen30_acc_mean_pp": 100.0 * statistics.mean(fc) / N_TEST,
                        "acc_gain_over_frozen30_pp": 100.0 * (statistics.mean(c) - statistics.mean(fc)) / N_TEST,
                        "train_acc_last_epoch_mean": statistics.mean(m["train_acc_last_epoch"] for m in ms),
                        "selected_epoch_mean": statistics.mean(m["best_val_epoch"] for m in ms),
                        "selected_last_epoch_count": sum(1 for m in ms if m["selected_last_epoch"]),
                        "selected_within_first30_count": sum(1 for m in ms if m["best_val_epoch"] <= PREFIX),
                        "unit_seconds_mean": statistics.mean(float(m["unit_seconds"] or 0.0) for m in ms)})

    results, changes = [], []
    for k, (cid, group, cond, a, b, decision, fid) in enumerate(CONTRASTS):
        d = [metrics[(cond, a, s)][1]["test_correct"] - metrics[(cond, b, s)][1]["test_correct"] for s in SEEDS]
        d30 = [frozen[frozen_uid(metrics[(cond, a, s)][0])]["test_correct"] - frozen[frozen_uid(metrics[(cond, b, s)][0])]["test_correct"]
               for s in SEEDS]
        st = int_stats(d, CONTRAST_SEED0 + k, N_TEST)
        results.append({"id": cid, "holm_group": group, "budget_epochs": EPOCHS, "condition": cond, "A": a, "B": b,
                        "decision_kind": decision, "frozen_id": fid, "direction": DIRECTION, **st,
                        "differences_count": ";".join(str(x) for x in d)})
        ch = [x - y for x, y in zip(d, d30)]
        lo, hi = mean_ci(ch, CHANGE_SEED0 + k)
        changes.append({"id": cid, "frozen_id": fid, "condition": cond, "A": a, "B": b,
                        "mean_diff_pp_100": 100.0 * statistics.mean(d) / N_TEST, "mean_diff_pp_frozen30": 100.0 * statistics.mean(d30) / N_TEST,
                        "budget_change_mean_pp": 100.0 * statistics.mean(ch) / N_TEST, "budget_change_ci95_low_pp": 100.0 * lo / N_TEST,
                        "budget_change_ci95_high_pp": 100.0 * hi / N_TEST, "seeds_change_positive": sum(1 for x in ch if x > 0),
                        "seeds_change_negative": sum(1 for x in ch if x < 0), "bootstrap_seed": CHANGE_SEED0 + k,
                        "bootstrap_resamples": BOOT_R, "frozen30_differences_count": ";".join(str(x) for x in d30),
                        "inference": "descriptive (no test, no label)"})
    for group in sorted({c[1] for c in CONTRASTS}):
        rows = [r for r in results if r["holm_group"] == group]
        for r, p in zip(rows, holm([x["p_two_sided"] for x in rows])):
            r["p_holm"] = p
            r["outcome"] = label(p, r)
    decisions = [{"criterion_id": r["id"], "holm_group": r["holm_group"], "budget_epochs": EPOCHS, "n": r["n"],
                  "mean_diff_pp": r["mean_diff_pp"], "ci_low_pp": r["ci95_low_pp"], "ci_high_pp": r["ci95_high_pp"], "holm_p": r["p_holm"],
                  "d_z": r["d_z"], "rank_biserial": r["rank_biserial"], "wilcoxon_method": r["wilcoxon_method"], "threshold": THRESHOLD,
                  "direction": DIRECTION, "outcome": r["outcome"], "discriminator": r["decision_kind"] == "rq2"} for r in results]
    agreement = []
    for r in results:
        f30 = frozen_rows[r["frozen_id"]]["outcome"]
        agreement.append({"id": r["id"], "frozen_id": r["frozen_id"], "frozen30_outcome": f30, "outcome_100": r["outcome"],
                          "agreement": "consistent" if f30 == r["outcome"] else "budget_dependent",
                          "frozen30_mean_diff_pp": float(frozen_rows[r["frozen_id"]]["mean_diff_pp"]), "mean_diff_pp_100": r["mean_diff_pp"]})
    differing = [a["id"] for a in agreement if a["agreement"] != "consistent"]
    rq2 = [r for r in results if r["decision_kind"] == "rq2"]
    sup = sum(1 for r in rq2 if r["outcome"] == "supported")
    rq2_frozen = [frozen_rows[r["frozen_id"]]["outcome"] for r in rq2]
    comp = [{"rule_id": "RQ2_robustness_e100", "inputs": ";".join(f"{r['id']}={r['outcome']}" for r in rq2), "supported": sup,
             "reverse": sum(1 for r in rq2 if r["outcome"] == "reverse"), "outcome": "robust" if sup >= 2 else "not_robust",
             "reverse_named": ";".join(r["condition"] for r in rq2 if r["outcome"] == "reverse"),
             "rule": "the A8 two-of-three summary over the three full-data DANN-LRF minus Naive-Branch contrasts, at 100 epochs; "
                     "reported beside the frozen RQ2_robustness verdict, never replacing it", "boundary": BOUNDARY},
            {"rule_id": "RQ2_robustness_frozen30", "inputs": ";".join(f"{r['frozen_id']}={o}" for r, o in zip(rq2, rq2_frozen)),
             "supported": rq2_frozen.count("supported"), "outcome": "robust" if rq2_frozen.count("supported") >= 2 else "not_robust",
             "rule": "the frozen verdict recomputed from the frozen labels, for reference", "boundary": BOUNDARY},
            {"rule_id": "budget_agreement", "inputs": ";".join(f"{a['id']}={a['frozen30_outcome']}|{a['outcome_100']}" for a in agreement),
             "outcome": "budget_dependent" if differing else "consistent", "differing": ";".join(differing),
             "rule": "DEC-NEURO-021: the frozen 30-epoch label is primary; where the 100-epoch label of the same contrast differs, the "
                     "statement is worded as budget-dependent and both labels are reported", "boundary": BOUNDARY},
            {"rule_id": "prefix_gate", "inputs": f"{sum(1 for g in gate if g['prefix_rows_equal'])}/{len(gate)} units",
             "outcome": "pass" if gate_pass else "determinism_finding",
             "rule": "epochs 1-30 of every 100-epoch history equal the frozen 30-epoch history row for row in every column",
             "boundary": BOUNDARY}]

    write_csv(out / "unit_table.csv", table)
    write_csv(out / "summary_by_cell.csv", summary)
    write_csv(out / "contrasts.csv", results)
    write_csv(out / "decision_inputs.csv", decisions)
    write_csv(out / "budget_agreement.csv", agreement)
    write_csv(out / "budget_change.csv", changes)
    write_csv(out / "prefix_gate.csv", gate)
    write_csv(out / "composite_decisions.csv", comp)
    ver = versions()
    run = {"script": "r1/aggregate_f1conv.py", "script_sha256": sha_file(Path(__file__)),
           "statistics_source": "r1/aggregate_r1.py int_stats, holm, label", "statistics_source_sha256": sha_file(ROOT / "r1" / "aggregate_r1.py"),
           "plan_sha256": sha_file(Path(args.plan)), "identity_sha256": sha_file(Path(args.identity)),
           "frozen_plan_sha256": sha_file(Path(args.frozen_plan)), "frozen_archive_sha256": archive_sha,
           "frozen_identity_sha256": sha_file(Path(args.frozen_identity)), "frozen_contrasts_sha256": sha_file(Path(args.frozen_contrasts)),
           "units": len(metrics), "data_hashes": data_hashes, "prefix_gate_pass": gate_pass,
           "lineage": ("epochs 1-30 of every 100-epoch history equal the frozen 30-epoch history row for row" if gate_pass else
                       "DETERMINISM FINDING: at least one prefix differs from its frozen history; the within-arm statistics stand, but the "
                       "budget comparison is not a same-trajectory comparison and is reported as such"),
           "contrasts": len(results), "contrast_bootstrap_seeds": {c[0]: CONTRAST_SEED0 + k for k, c in enumerate(CONTRASTS)},
           "cell_bootstrap_seeds": {f"{c}:{m}": CELL_SEED0 + k for k, (c, m) in enumerate(CELLS)},
           "change_bootstrap_seeds": {c[0]: CHANGE_SEED0 + k for k, c in enumerate(CONTRASTS)},
           "composite": {r["rule_id"]: r["outcome"] for r in comp}, "boundary": BOUNDARY,
           "budget_epochs": {"frozen_primary": PREFIX, "arm": EPOCHS}, "versions": ver, "versions_match_pinned": ver == PINNED}
    out.mkdir(parents=True, exist_ok=True)
    (out / "ANALYSIS_RUN.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8", newline="\n")
    cmp_ = run["composite"]
    print(f"units {len(metrics)}; prefix gate {cmp_['prefix_gate']}; contrasts {len(results)}; RQ2 at 100 epochs "
          f"{cmp_['RQ2_robustness_e100']} (frozen 30: {cmp_['RQ2_robustness_frozen30']}); budget agreement {cmp_['budget_agreement']}; "
          f"versions_match_pinned {run['versions_match_pinned']} -> {out}")
    return 0 if gate_pass else 3


# ---------------------------------------------------------------- selftest (synthetic data only)
def _history(rng, model: str, ds: str, seed: int, n: int, shift: int) -> list:
    """Synthetic history whose test_acc values are whole counts of 10,000 and whose val_loss has a known first minimum: a first
    minimum near epoch 22 and, for models after the first two, a lower second minimum after epoch 60, so that the 100-epoch
    selection differs from the 30-epoch one for part of the cells (non-zero budget changes)."""
    idx = ["dann_lrf", "naive_branch", "mlp_param", "dann_random"].index(model)
    rows, base = [], 5000 + 300 * idx + shift
    for e in range(1, n + 1):
        late = e > 60 and (idx >= 2 or seed % 3 == 0)
        cnt = base + 40 * min(e, 25) + int(rng.integers(0, 60)) + (25 * (idx + 1) + int(rng.integers(0, 40)) if late else 0)
        vl = 1.0 / min(e, 22) + 0.0005 * float(rng.random()) + (0.002 * (e - 22) if 22 < e <= 60 else 0.0)
        if late:
            vl = 1.0 / 22 - 0.004 * (idx + 1) + 0.0005 * float(rng.random())
        elif e > 60:
            vl = 1.0 / 22 + 0.08 + 0.0005 * float(rng.random())
        rows.append({"dataset": ds, "model_name": model, "seed": str(seed), "epoch": str(e),
                     "train_loss": repr(1.0 / e + 0.001 * float(rng.random())), "train_acc": repr(0.5 + 0.004 * min(e, 90)),
                     "val_loss": repr(vl), "val_acc": repr(cnt / 10000.0), "test_loss": repr(0.9 / e), "test_acc": repr(cnt / 10000.0)})
    return rows


def _write_unit(udir: Path, u: dict, rows: list, identity: dict, params: int, data_hashes: dict) -> None:
    udir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    (udir / "history.csv").write_bytes(buf.getvalue().encode("utf-8"))
    best = min(rows, key=lambda r: float(r["val_loss"]))
    summary = {"best_val_epoch": int(best["epoch"]), "test_acc_at_best_val": float(best["test_acc"]), "best_val_loss": float(best["val_loss"]),
               "test_loss_at_best_val": float(best["test_loss"]), "best_test_acc": max(float(r["test_acc"]) for r in rows),
               "final_test_acc": float(rows[-1]["test_acc"]), "trainable_params": params}
    result = {"schema_version": 1, "status": "completed", "unit_id": u["unit_id"], "unit_spec": unit_spec(u),
              "unit_spec_sha256": unit_spec_sha256(u), "history_file": "history.csv", "history_sha256": sha_file(udir / "history.csv"),
              "summary": summary, "model_info": {"effective_trainable_params": params}, "data_hashes": data_hashes,
              "timing_seconds": {"unit_total": 1.0}, "identity": identity}
    (udir / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")


def _synthetic(tmp: Path) -> dict:
    """Synthetic frozen plan/archive/receipt/identity/contrasts and a synthetic arm whose 30-epoch prefixes are exact."""
    import numpy as np

    from r1.r1_plan import f1conv_run, unit as plan_unit

    params = {"dann_lrf": 10634, "naive_branch": 10634, "mlp_param": 10606, "dann_random": 10634}
    dh = {ds: {"train_x": ds.upper() + "X", "train_y": ds.upper() + "Y", "test_x": "TX", "test_y": "TY"} for ds in DATASET_OF.values()}
    arm_units = f1conv_run()
    frozen_units = [plan_unit("main_grid", BASE_OF[u["condition"]], u["dataset"], 1.0, u["model"], u["seed"]) for u in arm_units]
    frozen_ident = {"freeze_id": "SYNTH-FROZEN", "run_id": "synthetic_frozen"}
    arm_ident = {"freeze_id": "SYNTH-ARM", "run_id": "synthetic_f1conv"}
    fz_root = tmp / "frozen_run" / "synthetic_frozen"
    arm_root = tmp / "arm_run"
    frozen_counts = {}
    for fu, au in zip(frozen_units, arm_units):
        rng = np.random.default_rng(int(hashlib.sha256(fu["unit_id"].encode("ascii")).hexdigest()[:12], 16))
        shift = {"fashion_full": 0, "kmnist_full": -400, "cifar_full": -2000}[fu["condition"]]
        rows100 = _history(rng, fu["model"], fu["dataset"], fu["seed"], EPOCHS, shift)
        _write_unit(fz_root / "units" / fu["unit_id"], fu, rows100[:PREFIX], frozen_ident, params[fu["model"]], dh[fu["dataset"]])
        _write_unit(arm_root / "units" / au["unit_id"], au, rows100, arm_ident, params[au["model"]], dh[au["dataset"]])
        best = min(rows100[:PREFIX], key=lambda r: float(r["val_loss"]))
        frozen_counts[(fu["condition"], fu["model"], fu["seed"])] = round(float(best["test_acc"]) * N_TEST)
    arc = tmp / "frozen.tar.gz"
    with tarfile.open(arc, "w:gz") as tar:
        tar.add(fz_root, arcname="synthetic_frozen")
    receipt = {"run_id": "synthetic_frozen", "archive": {"sha256": sha_file(arc)}}
    (tmp / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (tmp / "frozen_identity.json").write_text(json.dumps(frozen_ident), encoding="utf-8")
    (tmp / "identity.json").write_text(json.dumps(arm_ident), encoding="utf-8")
    fplan = {"schema_version": 1, "run_id": "synthetic_frozen", "units": frozen_units + [plan_unit("main_grid", "cifar_low02", "cifar10", 0.2, "dann_lrf", 0)]}
    (tmp / "frozen_plan.json").write_text(json.dumps(fplan, indent=2), encoding="utf-8")
    aplan = {"schema_version": 1, "run_id": "synthetic_f1conv", "units": arm_units}
    (tmp / "plan.json").write_text(json.dumps(aplan, indent=2), encoding="utf-8")
    # frozen labels computed with the frozen functions on the synthetic 30-epoch counts (the frozen seeds 20261003001...)
    frozen_seed = {"fashion_full": 20261003001, "kmnist_full": 20261003004, "cifar_full": 20261003007}
    rows = []
    for base in ("fashion_full", "kmnist_full", "cifar_full"):
        grp = []
        for j, b in enumerate(("mlp_param", "naive_branch", "dann_random")):
            d = [frozen_counts[(base, "dann_lrf", s)] - frozen_counts[(base, b, s)] for s in SEEDS]
            grp.append({"id": f"F1:{base}:dann_lrf-{b}", **int_stats(d, frozen_seed[base] + j, N_TEST)})
        for r, p in zip(grp, holm([x["p_two_sided"] for x in grp])):
            r["p_holm"] = p
            r["outcome"] = label(p, r)
        rows += grp
    write_csv(tmp / "frozen_contrasts.csv", rows)
    return {"root": arm_root, "plan": tmp / "plan.json", "identity": tmp / "identity.json", "frozen_plan": tmp / "frozen_plan.json",
            "frozen_archive": arc, "frozen_receipt": tmp / "receipt.json", "frozen_identity": tmp / "frozen_identity.json",
            "frozen_contrasts": tmp / "frozen_contrasts.csv", "frozen_counts": frozen_counts}


def _args(s: dict, out: Path, **over):
    a = argparse.Namespace(plan=str(s["plan"]), root=str(s["root"]), identity=str(s["identity"]), frozen_plan=str(s["frozen_plan"]),
                           frozen_archive=str(s["frozen_archive"]), frozen_receipt=str(s["frozen_receipt"]),
                           frozen_identity=str(s["frozen_identity"]), frozen_contrasts=str(s["frozen_contrasts"]), out=str(out))
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def selftest() -> int:
    from scipy.stats import wilcoxon

    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        s = _synthetic(tmp)
        out = tmp / "out"
        rc = _quiet(analyse, _args(s, out))
        check("analyse on the synthetic arm exits 0", rc == 0)
        con = {r["id"]: r for r in csv.DictReader(open(out / "contrasts.csv", encoding="utf-8", newline=""))}
        dec = {r["criterion_id"]: r for r in csv.DictReader(open(out / "decision_inputs.csv", encoding="utf-8", newline=""))}
        gate = list(csv.DictReader(open(out / "prefix_gate.csv", encoding="utf-8", newline="")))
        comp = {r["rule_id"]: r for r in csv.DictReader(open(out / "composite_decisions.csv", encoding="utf-8", newline=""))}
        agr = {r["id"]: r for r in csv.DictReader(open(out / "budget_agreement.csv", encoding="utf-8", newline=""))}
        chg = {r["id"]: r for r in csv.DictReader(open(out / "budget_change.csv", encoding="utf-8", newline=""))}
        summ = list(csv.DictReader(open(out / "summary_by_cell.csv", encoding="utf-8", newline="")))
        check("nine contrasts, three Holm families of three", len(con) == 9 and len({r["holm_group"] for r in con.values()}) == 3)
        check("240 prefix rows, all equal", len(gate) == 240 and all(g["prefix_rows_equal"] == "True" and g["prefix_bytes_equal"] == "True" for g in gate))
        check("twelve cells", len(summ) == 12)
        # independent recomputation of every contrast from the raw synthetic histories
        counts = {}
        for p in (s["root"] / "units").iterdir():
            rows = list(csv.DictReader(open(p / "history.csv", encoding="utf-8", newline="")))
            i = min(range(len(rows)), key=lambda j: (float(rows[j]["val_loss"]), j))
            cond, model, sd = p.name.split("__")
            counts[(cond, model, int(sd[1:]))] = round(float(rows[i]["test_acc"]) * N_TEST)
        for k, (cid, group, cond, a, b, decision, fid) in enumerate(CONTRASTS):
            d = [counts[(cond, a, x)] - counts[(cond, b, x)] for x in SEEDS]
            nz = [x for x in d if x != 0]
            method = "exact" if (len(nz) == len(d) and len({abs(x) for x in d}) == len(d)) else "approx"
            p = float(wilcoxon(d, zero_method="wilcox", correction=False, alternative="two-sided", method=method).pvalue) if nz else 1.0
            check(f"{cid}: mean difference", math.isclose(float(con[cid]["mean_diff_count"]), statistics.mean(d), rel_tol=0, abs_tol=1e-9))
            check(f"{cid}: Wilcoxon p ({method})", math.isclose(float(con[cid]["p_two_sided"]), p, rel_tol=1e-12, abs_tol=1e-15))
            check(f"{cid}: differences string", con[cid]["differences_count"] == ";".join(map(str, d)))
            check(f"{cid}: bootstrap seed", int(con[cid]["bootstrap_seed"]) == CONTRAST_SEED0 + k)
            d30 = [s["frozen_counts"][(BASE_OF[cond], a, x)] - s["frozen_counts"][(BASE_OF[cond], b, x)] for x in SEEDS]
            check(f"{cid}: budget change mean", math.isclose(float(chg[cid]["budget_change_mean_pp"]),
                                                              100.0 * (statistics.mean(d) - statistics.mean(d30)) / N_TEST, abs_tol=1e-9))
        for g in sorted({c[1] for c in CONTRASTS}):
            ids = [c[0] for c in CONTRASTS if c[1] == g]
            ps = [float(con[i]["p_two_sided"]) for i in ids]
            order = sorted(range(3), key=lambda i: ps[i])
            adj, run_ = [0.0] * 3, 0.0
            for r_, i in enumerate(order):
                run_ = max(run_, min(1.0, (3 - r_) * ps[i]))
                adj[i] = run_
            for i, a_ in zip(ids, adj):
                check(f"{i}: Holm", math.isclose(float(dec[i]["holm_p"]), a_, rel_tol=1e-12, abs_tol=1e-15))
                lo, hi = float(con[i]["ci95_low_count"]), float(con[i]["ci95_high_count"])
                want = "supported" if a_ < 0.05 and lo > 0 else "reverse" if a_ < 0.05 and hi < 0 else "not_supported"
                check(f"{i}: A8 label", dec[i]["outcome"] == want)
        frozen_lab = {r["id"]: r["outcome"] for r in csv.DictReader(open(s["frozen_contrasts"], encoding="utf-8", newline=""))}
        for cid, *_rest in CONTRASTS:
            fid = _rest[-1]
            want = "consistent" if frozen_lab[fid] == dec[cid]["outcome"] else "budget_dependent"
            check(f"{cid}: budget agreement", agr[cid]["agreement"] == want)
        check("the synthetic data exercise a moved selection (some budget changes are non-zero)",
              sum(1 for r in chg.values() if float(r["budget_change_mean_pp"]) != 0.0) >= 3)
        check("some cells select an epoch beyond the 30-epoch prefix", any(float(r["selected_epoch_mean"]) > PREFIX for r in summ))
        check("prefix gate composite passes", comp["prefix_gate"]["outcome"] == "pass")
        run = json.loads((out / "ANALYSIS_RUN.json").read_text(encoding="utf-8"))
        check("ANALYSIS_RUN records the prefix gate", run["prefix_gate_pass"] is True and run["units"] == 240)
        check("bootstrap seeds of contrasts, cells and changes distinct",
              len(set(run["contrast_bootstrap_seeds"].values()) | set(run["cell_bootstrap_seeds"].values()) | set(run["change_bootstrap_seeds"].values())) == 30)

        # negative controls (each on a fresh copy)
        def variant(name, mutate, want_rc):
            vt = tmp / f"v_{name}"
            shutil.copytree(tmp, vt, ignore=shutil.ignore_patterns("v_*", "out*"))
            sv = {k: (vt / Path(v).relative_to(tmp) if isinstance(v, Path) else v) for k, v in s.items()}
            mutate(sv)
            rc_ = _quiet(analyse, _args(sv, vt / "outv"))
            check(f"negative control {name}: exit {want_rc}", rc_ == want_rc)
            return vt

        def drop_unit(sv):
            (sv["root"] / "units" / "kmnist_full_e100__naive_branch__s07" / "result.json").unlink()

        def extra_unit(sv):
            shutil.copytree(sv["root"] / "units" / "kmnist_full_e100__naive_branch__s07", sv["root"] / "units" / "kmnist_full_e100__naive_branch__s99")

        def wrong_params(sv):
            p = sv["root"] / "units" / "cifar_full_e100__dann_lrf__s03" / "result.json"
            r = json.loads(p.read_text(encoding="utf-8"))
            r["model_info"]["effective_trainable_params"] = 10635
            p.write_text(json.dumps(r), encoding="utf-8")

        def wrong_data(sv):
            p = sv["root"] / "units" / "fashion_full_e100__mlp_param__s11" / "result.json"
            r = json.loads(p.read_text(encoding="utf-8"))
            r["data_hashes"]["train_x"] = "OTHER"
            p.write_text(json.dumps(r), encoding="utf-8")

        def broken_prefix(sv):
            ud = sv["root"] / "units" / "cifar_full_e100__naive_branch__s05"
            r = json.loads((ud / "result.json").read_text(encoding="utf-8"))
            rows = list(csv.DictReader(open(ud / "history.csv", encoding="utf-8", newline="")))
            rows[3]["train_loss"] = repr(float(rows[3]["train_loss"]) + 1e-9)  # epoch 4 of 30: not the selected epoch, summary unchanged
            buf = io.StringIO(newline="")
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
            (ud / "history.csv").write_bytes(buf.getvalue().encode("utf-8"))
            r["history_sha256"] = sha_file(ud / "history.csv")
            (ud / "result.json").write_text(json.dumps(r), encoding="utf-8")

        def changed_lr(sv):
            p = json.loads(Path(sv["plan"]).read_text(encoding="utf-8"))
            p["units"][5]["lr"] = 0.003
            Path(sv["plan"]).write_text(json.dumps(p), encoding="utf-8")

        def bad_receipt(sv):
            Path(sv["frozen_receipt"]).write_text(json.dumps({"run_id": "synthetic_frozen", "archive": {"sha256": "0" * 64}}), encoding="utf-8")

        def missing_frozen_label(sv):
            rows = [r for r in csv.DictReader(open(sv["frozen_contrasts"], encoding="utf-8", newline="")) if r["id"] != "F1:cifar_full:dann_lrf-naive_branch"]
            write_csv(Path(sv["frozen_contrasts"]), rows)

        variant("missing unit", drop_unit, 2)
        variant("unplanned unit folder", extra_unit, 2)
        variant("parameter count", wrong_params, 2)
        variant("data hashes", wrong_data, 2)
        vb = variant("broken prefix", broken_prefix, 3)
        gate_b = {r["unit_id"]: r for r in csv.DictReader(open(vb / "outv" / "prefix_gate.csv", encoding="utf-8", newline=""))}
        check("broken prefix is named in prefix_gate.csv", gate_b["cifar_full_e100__naive_branch__s05"]["prefix_rows_equal"] == "False"
              and sum(1 for r in gate_b.values() if r["prefix_rows_equal"] == "True") == 239)
        variant("changed hyper-parameter in the plan", changed_lr, 2)
        variant("frozen archive does not match its receipt", bad_receipt, 2)
        variant("frozen label missing", missing_frozen_label, 2)

    failed = [n for n, ok in checks if not ok]
    for n in failed:
        print("FAILED:", n)
    print(f"selftest: {len(checks)} checks, {len(failed)} failed")
    ver = versions()
    print(f"versions {ver}; pinned {PINNED}; match {ver == PINNED}")
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse")
    for k in ("plan", "root", "identity", "frozen-plan", "frozen-archive", "frozen-receipt", "frozen-identity", "frozen-contrasts", "out"):
        a.add_argument(f"--{k}", required=True)
    sub.add_parser("selftest")
    args = ap.parse_args()
    return analyse(args) if args.cmd == "analyse" else selftest()


if __name__ == "__main__":
    raise SystemExit(main())
