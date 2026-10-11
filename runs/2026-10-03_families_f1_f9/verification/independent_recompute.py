"""Independent recomputation of the frozen R1 decision inputs (second method; shares no code with aggregate_r1.py).

    python independent_recompute.py <units_root> <plan.json> <R1_ANALYSIS_MANIFEST.json> <processed_outputs dir> <out.json>

Re-derives every unit's correct-count from its raw history (first epoch of minimum val_loss), rebuilds every
contrast from the manifest, and recomputes: the mean difference, an EXACT two-sided Wilcoxon signed-rank p-value
by dynamic programming over rank sums when there are no zeros and no tied |d| (otherwise the normal approximation
with the tie-corrected variance, no continuity correction), Holm within each holm_group, the paired percentile
bootstrap CI and the decision label. Every value is compared with processed_outputs/contrasts.csv and
decision_inputs.csv. Internal invariants are asserted; a failed invariant prints TOOL FAULT and exits 3.
Exit 0 when every contrast agrees, 2 when any disagrees.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

units_root, plan_path, man_path, proc, out_path = (Path(a) for a in sys.argv[1:6])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
man = json.loads(man_path.read_text(encoding="utf-8"))
n_test = int(man["n_test"])

# 1. counts from raw histories
counts = {}
for u in plan["units"]:
    udir = units_root / "units" / u["unit_id"]
    res = json.loads((udir / "result.json").read_text(encoding="utf-8"))
    with open(udir / res["history_file"], encoding="utf-8", newline="") as fh:
        hist = list(csv.DictReader(fh))
    if not hist:
        print("TOOL FAULT: empty history", u["unit_id"]); sys.exit(3)
    best_i = min(range(len(hist)), key=lambda i: (float(hist[i]["val_loss"]), i))
    acc = float(hist[best_i]["test_acc"])
    c = round(acc * n_test)
    if abs(acc * n_test - c) > 1e-6:
        print("TOOL FAULT: non-integer count", u["unit_id"], acc); sys.exit(3)
    key = (u["family"], u["condition"], u["model"], int(u["seed"]))
    if key in counts:
        print("TOOL FAULT: duplicate unit key", key); sys.exit(3)
    counts[key] = c
if len(counts) != len(plan["units"]):
    print("TOOL FAULT: unit count mismatch"); sys.exit(3)


def cell(c, s):
    return counts[(c["family"], c["condition"], c["model"], int(s))]


def exact_wilcoxon_p(d):
    """Exact two-sided p for distinct nonzero |d| (ranks 1..n): DP over the distribution of W+."""
    n = len(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0] * n
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    total = n * (n + 1) // 2
    dist = [0] * (total + 1)
    dist[0] = 1
    for r in range(1, n + 1):
        for s in range(total, r - 1, -1):
            dist[s] += dist[s - r]
    ways = 2 ** n
    assert sum(dist) == ways, "TOOL FAULT: DP mass"
    w = min(w_plus, total - w_plus)
    p = 2.0 * sum(dist[: w + 1]) / ways
    return min(1.0, p), float(min(w_plus, total - w_plus)), "exact"


def approx_wilcoxon_p(d):
    """Normal approximation, zeros dropped (wilcox), average ranks for ties, tie-corrected variance, no correction."""
    nz = [x for x in d if x != 0]
    n = len(nz)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, nz) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, nz) if x < 0)
    t_stat = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    z = (t_stat - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return min(1.0, p), float(t_stat), "approx"


def stats(d, seed):
    d = [int(x) for x in d]
    n = len(d)
    arr = np.asarray(d, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    means = arr[rng.integers(0, n, size=(10_000, n))].mean(axis=1)
    lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    nz = [x for x in d if x != 0]
    if not nz:
        p, w, method = 1.0, None, "all_zero"
    elif len(nz) == n and len({abs(x) for x in nz}) == n and n <= 50:
        p, w, method = exact_wilcoxon_p(d)
    else:
        p, w, method = approx_wilcoxon_p(d)
    return {"n": n, "mean_diff_pp": 100.0 * sum(d) / n / n_test, "ci_low_pp": 100.0 * lo / n_test,
            "ci_high_pp": 100.0 * hi / n_test, "ci_low_count": lo, "ci_high_count": hi, "p": p, "method": method}


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


res = {}
for c in man["contrasts"]:
    if c["kind"] == "did":
        d = [(cell(c["A"]["minuend"], s) - cell(c["A"]["subtrahend"], s)) - (cell(c["B"]["minuend"], s) - cell(c["B"]["subtrahend"], s))
             for s in c["seeds"]]
    else:
        d = [cell(c["A"], s) - cell(c["B"], s) for s in c["seeds"]]
    res[c["id"]] = {"holm_group": c["holm_group"], **stats(d, c["bootstrap_seed"])}
groups = {}
for cid, r in res.items():
    groups.setdefault(r["holm_group"], []).append(cid)
for ids in groups.values():
    for cid, ph in zip(ids, holm([res[i]["p"] for i in ids])):
        r = res[cid]
        r["p_holm"] = ph
        if r["method"] == "all_zero":
            r["label"] = "no_difference"
        elif ph < 0.05 and r["ci_low_count"] > 0:
            r["label"] = "supported"
        elif ph < 0.05 and r["ci_high_count"] < 0:
            r["label"] = "reverse"
        else:
            r["label"] = "not_supported"

with open(proc / "decision_inputs.csv", encoding="utf-8") as fh:
    frozen = {r["criterion_id"]: r for r in csv.DictReader(fh)}
with open(proc / "contrasts.csv", encoding="utf-8") as fh:
    frozen_c = {r["id"]: r for r in csv.DictReader(fh)}
if set(frozen) != set(res) or set(frozen_c) != set(res):
    print("TOOL FAULT: contrast id sets differ"); sys.exit(3)
rows, disagree = [], 0
for cid, r in res.items():
    f, fc = frozen[cid], frozen_c[cid]
    checks = {
        "mean_diff": abs(float(f["mean_diff_pp"]) - r["mean_diff_pp"]) < 1e-9,
        "ci": abs(float(f["ci_low_pp"]) - r["ci_low_pp"]) < 1e-9 and abs(float(f["ci_high_pp"]) - r["ci_high_pp"]) < 1e-9,
        "method": fc["wilcoxon_method"] == r["method"],
        "p_raw": abs(float(fc["p_two_sided"]) - r["p"]) <= 1e-9 + 1e-6 * r["p"],
        "p_holm": abs(float(f["holm_p"]) - r["p_holm"]) <= 1e-9 + 1e-6 * r["p_holm"],
        "label": f["outcome"] == r["label"],
    }
    ok = all(checks.values())
    disagree += not ok
    rows.append({"id": cid, "agree": ok, **{k: v for k, v in checks.items()}, "independent": {k: r[k] for k in
                 ("n", "mean_diff_pp", "ci_low_pp", "ci_high_pp", "p", "p_holm", "method", "label")},
                 "frozen": {"mean_diff_pp": float(f["mean_diff_pp"]), "p_two_sided": float(fc["p_two_sided"]),
                            "holm_p": float(f["holm_p"]), "method": fc["wilcoxon_method"], "outcome": f["outcome"]}})
labels = {}
for r in res.values():
    labels[r["label"]] = labels.get(r["label"], 0) + 1
assert sum(labels.values()) == len(res), "TOOL FAULT: label tally"
doc = {"schema_version": 1, "method": "independent re-derivation from raw histories; exact Wilcoxon by DP; own Holm and labels",
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
       "units": len(counts), "contrasts": len(res), "disagreements": disagree, "label_tally": labels, "rows": rows}
out_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"units {len(counts)}; contrasts {len(res)}; disagreements {disagree}; labels {labels}")
for r in rows:
    if not r["agree"]:
        print("DISAGREE", r["id"], {k: r[k] for k in ("mean_diff", "ci", "method", "p_raw", "p_holm", "label")}, r["independent"], r["frozen"])
sys.exit(0 if disagree == 0 else 2)
