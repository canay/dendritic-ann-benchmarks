"""Aggregate validated R1 unit results exactly as the frozen analysis manifest prescribes (no model code).

    python r1/aggregate_r1.py --plan <plan.json> --units-root <extracted run folder> --manifest <R1_ANALYSIS_MANIFEST.json>
        --out <processed dir> [--identity-file <IDENTITY.json>] [--r0-paired <paper_package/derived/paired_tests_validation_selected.csv>]

Fails closed (exit 2) when any planned unit is missing or invalid. The primary metric is RE-DERIVED from each history
(first epoch of minimum val_loss) and analysed as the integer number of correct test images (review PCR-006, PCR-011).
Outputs: unit_table.csv, summary_by_cell.csv, contrasts.csv, decision_inputs.csv, composite_decisions.csv,
descriptive.csv, replication_r0_vs_r1.csv (when --r0-paired is given), a4_checks.csv, ANALYSIS_RUN.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1.r1_runner import load_plan, validate_result  # noqa: E402

BOOT_R = 10_000
METRIC_COLUMNS = ("train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running, m = [0.0] * len(pvals), 0.0, len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial on integer differences: average ranks over |d|, zeros dropped."""
    nz = [int(x) for x in d if int(x) != 0]
    if not nz:
        return 0.0
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks = [0.0] * len(nz)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    wp = sum(r for r, x in zip(ranks, nz) if x > 0)
    wn = sum(r for r, x in zip(ranks, nz) if x < 0)
    return (wp - wn) / (wp + wn)


def int_stats(d, boot_seed: int, n_test: int, test: bool = True) -> dict:
    import numpy as np
    from scipy.stats import wilcoxon

    d = np.asarray([int(x) for x in d], dtype=np.int64)
    n = len(d)
    rng = np.random.default_rng(int(boot_seed))
    means = d[rng.integers(0, n, size=(BOOT_R, n))].mean(axis=1)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    out = {"n": n, "mean_diff_count": float(d.mean()), "ci95_low_count": float(np.quantile(means, 0.025)),
           "ci95_high_count": float(np.quantile(means, 0.975)), "d_z": (float(d.mean()) / sd) if sd > 0 else "undefined",
           "rank_biserial": rank_biserial(d), "bootstrap_seed": int(boot_seed), "bootstrap_resamples": BOOT_R}
    for key in ("mean_diff", "ci95_low", "ci95_high"):
        out[key + "_pp"] = 100.0 * out[key + "_count"] / n_test
    if test:
        nz = d[d != 0]
        if len(nz) == 0:
            out.update(wilcoxon_W="", p_two_sided=1.0, wilcoxon_method="all_zero")
        else:
            exact = len(nz) == n and len(np.unique(np.abs(nz))) == len(nz) and n <= 50
            method = "exact" if exact else "approx"
            res = wilcoxon(d, zero_method="wilcox", correction=False, alternative="two-sided", method=method)
            out.update(wilcoxon_W=float(res.statistic), p_two_sided=float(res.pvalue), wilcoxon_method=method)
    return out


def label(p_holm: float, row: dict) -> str:
    if row.get("wilcoxon_method") == "all_zero":
        return "no_difference"
    if p_holm < 0.05 and row["ci95_low_count"] > 0:
        return "supported"
    if p_holm < 0.05 and row["ci95_high_count"] < 0:
        return "reverse"
    return "not_supported"


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_history(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--units-root", required=True, help="folder that contains units/<unit_id>/")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity-file", default=None)
    ap.add_argument("--r0-paired", default=None)
    args = ap.parse_args()
    plan = load_plan(Path(args.plan))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    identity = json.loads(Path(args.identity_file).read_text(encoding="utf-8")) if args.identity_file else None
    n_test = int(manifest["n_test"])
    root, out = Path(args.units_root), Path(args.out)

    rows, bad, counts, hist = [], [], {}, {}
    for u in plan["units"]:
        udir = root / "units" / u["unit_id"]
        reason = validate_result(u, udir, identity)
        if reason is not None:
            bad.append((u["unit_id"], reason))
            continue
        r = json.loads((udir / "result.json").read_text(encoding="utf-8"))
        h = read_history(udir / r["history_file"])
        best = min(h, key=lambda x: float(x["val_loss"]))
        acc = float(best["test_acc"])
        cnt = round(acc * n_test)
        if abs(acc * n_test - cnt) > 1e-6:
            bad.append((u["unit_id"], f"accuracy {acc!r} is not a whole number of the {n_test} test images"))
            continue
        key = (u["family"], u["condition"], u["model"], int(u["seed"]))
        counts[key], hist[key] = cnt, udir / r["history_file"]
        mi = r.get("model_info") or {}
        rows.append({"unit_id": u["unit_id"], "family": u["family"], "condition": u["condition"], "model": u["model"],
                     "seed": u["seed"], "dataset": u["dataset"], "subset_fraction": u["subset_fraction"], "lr": u["lr"],
                     "extra": json.dumps(u.get("extra") or {}, sort_keys=True), "best_val_epoch": int(best["epoch"]),
                     "test_correct_at_best_val": cnt, "test_acc_at_best_val": acc,
                     "effective_params": mi.get("effective_trainable_params", r["summary"]["trainable_params"]),
                     "unit_seconds": r["timing_seconds"]["unit_total"], "history_sha256": r["history_sha256"]})
    if bad:
        for uid, reason in bad[:20]:
            print("INVALID", uid, reason)
        print(f"FAIL: {len(bad)} of {len(plan['units'])} planned units are missing or invalid; nothing aggregated")
        return 2
    write_csv(out / "unit_table.csv", rows)

    def get(c: dict, s: int) -> int:
        return counts[(c["family"], c["condition"], c["model"], s)]

    groups = {}
    for (fam, cond, model, s), v in counts.items():
        groups.setdefault((fam, cond, model), []).append(v / n_test)
    write_csv(out / "summary_by_cell.csv", [{"family": f, "condition": c, "model": m, "n_seeds": len(v), "acc_mean": statistics.mean(v),
                                              "acc_sd": statistics.stdev(v) if len(v) > 1 else 0.0, "acc_min": min(v), "acc_max": max(v)}
                                             for (f, c, m), v in sorted(groups.items())])

    # inferential contrasts, Holm within holm_group
    results = []
    for c in manifest["contrasts"]:
        if c["kind"] == "did":
            a = [get(c["A"]["minuend"], s) - get(c["A"]["subtrahend"], s) for s in c["seeds"]]
            b = [get(c["B"]["minuend"], s) - get(c["B"]["subtrahend"], s) for s in c["seeds"]]
        else:
            a = [get(c["A"], s) for s in c["seeds"]]
            b = [get(c["B"], s) for s in c["seeds"]]
        st = int_stats([x - y for x, y in zip(a, b)], c["bootstrap_seed"], n_test)
        results.append({"id": c["id"], "holm_group": c["holm_group"], "decision_kind": c["decision"], **st})
    by_group = {}
    for r in results:
        by_group.setdefault(r["holm_group"], []).append(r)
    for grp in by_group.values():
        for r, p in zip(grp, holm([x["p_two_sided"] for x in grp])):
            r["p_holm"] = p
            r["outcome"] = label(p, r)
    write_csv(out / "contrasts.csv", results)
    by_id = {r["id"]: r for r in results}
    decisions = [{"criterion_id": r["id"], "holm_group": r["holm_group"], "n": r["n"], "mean_diff_pp": r["mean_diff_pp"],
                  "ci_low_pp": r["ci95_low_pp"], "ci_high_pp": r["ci95_high_pp"], "holm_p": r["p_holm"], "d_z": r["d_z"],
                  "rank_biserial": r["rank_biserial"], "threshold": manifest["decision_rule"]["supported"] + " (reverse: " + manifest["decision_rule"]["reverse"] + ")",
                  "outcome": r["outcome"], "discriminator": r["decision_kind"] in ("rq2", "rq2_controlled", "transfer", "locality")}
                 for r in results]
    write_csv(out / "decision_inputs.csv", decisions)

    # composite rules (manifest composite_rules; logic mirrors the rule text)
    comp = []
    rq2 = [by_id[i]["outcome"] for i in manifest["composite_rules"][0]["inputs"]]
    comp.append({"rule_id": "RQ2_robustness", "inputs": ";".join(rq2), "supported": rq2.count("supported"), "reverse": rq2.count("reverse"),
                 "outcome": "robust" if rq2.count("supported") >= 2 else "not_robust"})
    for f1_id, f8_id in manifest["composite_rules"][1]["inputs"]:
        a, b = by_id[f1_id]["outcome"], by_id[f8_id]["outcome"]
        verdict = ("not_attributed_initialisation_not_excluded" if a == "supported" and b != "supported"
                   else "controlled_estimate_reported" if b == "supported" and a != "supported"
                   else "consistent_supported" if a == b == "supported" else "consistent_not_supported" if a != "supported" and b != "supported" else "check")
        comp.append({"rule_id": "F8b_vs_F1", "inputs": f"{f1_id}={a};{f8_id}={b}", "outcome": verdict})
    tr = [by_id[i]["outcome"] for i in manifest["composite_rules"][2]["inputs"]]
    comp.append({"rule_id": "F9_transfer", "inputs": ";".join(tr), "supported": tr.count("supported"), "reverse": tr.count("reverse"),
                 "outcome": "transfer_observed" if tr.count("supported") == 2 else "dataset_specific" if tr.count("supported") == 1 else "not_observed"})
    for i in manifest["composite_rules"][3]["inputs"]:
        r = by_id[i]
        ok = r["p_holm"] < 0.05 and r["ci95_high_count"] < 0
        comp.append({"rule_id": "F5_locality", "inputs": i, "outcome": "attenuation_supported" if ok else "measured_change_reported",
                     "mean_diff_pp": r["mean_diff_pp"], "ci_low_pp": r["ci95_low_pp"], "ci_high_pp": r["ci95_high_pp"]})
    write_csv(out / "composite_decisions.csv", comp)

    # descriptive families (no tests)
    desc = []
    for d in manifest["descriptive"]:
        if d["kind"] == "cell":
            v = [get(d["cell"], s) / n_test for s in d["seeds"]]
            desc.append({"id": d["id"], "n": len(v), "mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0})
        elif d["kind"] == "pair":
            diff = [get(d["A"], s) - get(d["B"], s) for s in d["seeds"]]
            desc.append({"id": d["id"], **int_stats(diff, d["bootstrap_seed"], n_test, test=False)})
        elif d["kind"] == "sd_across_values":
            a = [get(d["A"], s) / n_test for s in d["seeds"]]
            b = [get(d["B"], s) / n_test for s in d["seeds"]]
            desc.append({"id": d["id"], "n": len(a), "sd_A": statistics.stdev(a), "sd_B": statistics.stdev(b),
                         "sd_A_minus_B": statistics.stdev([x - y for x, y in zip(a, b)])})
    write_csv(out / "descriptive.csv", desc)

    # A4 identity checks (PCR-010)
    a4 = []
    for pair in manifest["a4_checks"]["pairs"]:
        l, r = pair["left"], pair["right"]
        hl, hr = read_history(hist[(l["family"], l["condition"], l["model"], l["seed"])]), read_history(hist[(r["family"], r["condition"], r["model"], r["seed"])])
        same = len(hl) == len(hr) and all(x["epoch"] == y["epoch"] and x["dataset"] == y["dataset"] and x["seed"] == y["seed"]
                                          and all(x[k] == y[k] for k in METRIC_COLUMNS) for x, y in zip(hl, hr))
        names_ok = all(x["model_name"] == l["model"] for x in hl) and all(y["model_name"] == r["model"] for y in hr)
        a4.append({"id": pair["id"], "numeric_equal": same, "model_names_as_expected": names_ok})
    write_csv(out / "a4_checks.csv", a4)
    a4_pass = all(x["numeric_equal"] and x["model_names_as_expected"] for x in a4)

    # replication against the r0 contrast set (descriptive)
    rep = []
    if args.r0_paired:
        with open(args.r0_paired, encoding="utf-8") as f:
            r0 = list(csv.DictReader(f))
        seeds = manifest["replication"]["seeds"]
        stats_by_cond = {}
        for k, x in enumerate(r0):
            diff = [counts[("main_grid", x["condition_key"], x["A"], s)] - counts[("main_grid", x["condition_key"], x["B"], s)] for s in seeds]
            st = int_stats(diff, 20261003900 + k, n_test)
            stats_by_cond.setdefault(x["condition_key"], []).append((x, st))
        for cond, items in stats_by_cond.items():
            for (x, st), p in zip(items, holm([s["p_two_sided"] for _, s in items])):
                r0_sign = (float(x["mean_diff"]) > 0) - (float(x["mean_diff"]) < 0)
                r1_sign = (st["mean_diff_count"] > 0) - (st["mean_diff_count"] < 0)
                r0_dec = float(x["p_holm_per_condition"]) < 0.05
                rep.append({"condition": cond, "A": x["A"], "B": x["B"], "r0_mean_diff_pp": 100 * float(x["mean_diff"]),
                            "r0_p_holm": float(x["p_holm_per_condition"]), "r1_mean_diff_pp_seeds0_9": st["mean_diff_pp"], "r1_p_holm_seeds0_9": p,
                            "sign_agrees": r0_sign == r1_sign, "holm_decision_agrees": r0_dec == (p < 0.05)})
        write_csv(out / "replication_r0_vs_r1.csv", rep)

    import numpy
    import scipy

    versions = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}
    run = {"manifest": args.manifest, "manifest_sha256": sha_file(Path(args.manifest)), "aggregate_sha256": sha_file(Path(__file__)),
           "plan": args.plan, "plan_sha256": sha_file(Path(args.plan)), "units": len(rows), "versions": versions,
           "versions_match_pinned": versions == manifest["statistics"]["pinned_versions"], "a4_pass": a4_pass,
           "identity_checked": identity is not None}
    (out / "ANALYSIS_RUN.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"aggregated {len(rows)} units; contrasts {len(results)}; a4_pass {a4_pass}; versions_match_pinned {run['versions_match_pinned']} -> {out}")
    return 0 if a4_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
