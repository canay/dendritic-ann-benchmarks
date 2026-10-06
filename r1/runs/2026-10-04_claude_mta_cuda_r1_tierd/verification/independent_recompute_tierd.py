"""Independent recomputation of the tier D decision inputs (MC-NEURO-R1-003, protocol Section 10, A14-A17).

Second method: shares no code with r1/aggregate_tierd.py or r1/aggregate_r1.py (nothing is imported from r1/). The contrast,
cell and seed definitions below are written from the protocol text, not copied from the analysis script.

    python independent_recompute_tierd.py <run root holding units/> <plan_tierd_run.json> <processed_outputs dir>
        <frozen outputs archive.tar.gz> <out.json>

1. Every planned unit's correct count is re-derived from its own history.csv (first epoch of minimum val_loss; A9) and must be a
   whole number of the 10,000 test images; the history must have exactly the planned number of epochs.
2. The fourteen contrasts (seven at 30 epochs, seven at 100 epochs; A16 e, A17) are rebuilt from the counts: mean difference,
   two-sided Wilcoxon signed-rank p (EXACT by dynamic programming over the distribution of W+ when no difference is zero, no
   absolute difference is tied and n <= 50; otherwise the normal approximation with average ranks, tie-corrected variance and no
   continuity correction), Holm within each of the four families, the 95 % percentile bootstrap CI with R = 10,000 and the
   documented generator seed, d_z, and the A8 label.
3. The 22 cells are rebuilt: mean accuracy, SD, bootstrap CI with the documented cell seeds, the number of seeds selected at the
   last epoch.
4. The composites (D1 transfer and D2 increment at each budget; the budget agreement of rule vi) are rebuilt.
5. The nine lineage anchors are compared byte for byte with the histories of their frozen units, read from the frozen archive.
Every value is compared with the analysis outputs (contrasts.csv, decision_inputs.csv, summary_by_cell.csv,
composite_decisions.csv, anchors.csv). Internal invariants are asserted; a failed invariant prints TOOL FAULT and exits 3.
Exit 0 when everything agrees, 2 when anything disagrees.
Written before any result of the run existed. Operation neucom-r1-tierd-protocol-lock-20261005 (Cowork-Claude,
claude-opus-5-5, max).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path

import numpy as np

N_TEST = 10_000
SEEDS = list(range(20))
R_BOOT = 10_000
HEADS = ["tf_stem_dann_lrf", "tf_stem_naive_branch", "tf_stem_mlp"]
D2 = ["dann_lrf", "naive_branch", "mlp_param", "dann_random", "vann_same"]


def fault(msg: str) -> None:
    print("TOOL FAULT:", msg)
    sys.exit(3)


# ---- definitions written from protocol Section 10 (A16 e and A17)
def contrast_defs():
    out = []
    for suffix, d1g, d2g in (("", "D1", "D2"), ("_e100", "D1_e100", "D2_e100")):
        for cond in ("fashion_full" + suffix, "cifar_full" + suffix):
            for b in ("tf_stem_naive_branch", "tf_stem_mlp"):
                out.append({"id": f"{d1g}:{cond}:tf_stem_dann_lrf-{b}", "group": d1g, "cond": cond, "A": "tf_stem_dann_lrf", "B": b,
                            "transfer": b == "tf_stem_naive_branch"})
        for b in ("naive_branch", "mlp_param", "dann_random"):
            cond = "cifar100_full" + suffix
            out.append({"id": f"{d2g}:{cond}:dann_lrf-{b}", "group": d2g, "cond": cond, "A": "dann_lrf", "B": b, "increment": b == "naive_branch"})
    for k, c in enumerate(out):  # k = 0-6 primary, 7-13 the A17 arm, in the protocol's order
        c["boot_seed"] = 20261004300 + k
    return out


def cell_defs():
    out = []
    for suffix in ("", "_e100"):
        out += [("fashion_full" + suffix, m) for m in HEADS] + [("cifar_full" + suffix, m) for m in HEADS] + \
               [("cifar100_full" + suffix, m) for m in D2]
    return [(c, m, 20261004400 + k) for k, (c, m) in enumerate(out)]


# ---- statistics
def exact_p(d):
    n = len(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0] * n
    for r, i in enumerate(order, start=1):
        rank[i] = r
    w_plus = sum(r for r, x in zip(rank, d) if x > 0)
    total = n * (n + 1) // 2
    dist = [1] + [0] * total
    for r in range(1, n + 1):
        for s in range(total, r - 1, -1):
            dist[s] += dist[s - r]
    if sum(dist) != 2 ** n:
        fault("DP mass")
    w = min(w_plus, total - w_plus)
    return min(1.0, 2.0 * sum(dist[: w + 1]) / 2 ** n)


def approx_p(d):
    nz = [x for x in d if x != 0]
    n = len(nz)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks, tie_term, i = [0.0] * n, 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, nz) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, nz) if x < 0)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    z = (min(w_plus, w_minus) - n * (n + 1) / 4.0) / math.sqrt(var)
    return min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))


def boot_ci(values, seed):
    arr = np.asarray([int(v) for v in values], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    means = arr[rng.integers(0, len(arr), size=(R_BOOT, len(arr)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main() -> int:
    root, plan_path, proc, frozen_arc, out_path = (Path(a) for a in sys.argv[1:6])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    units = plan["units"]
    if len(units) != 449 or len({u["unit_id"] for u in units}) != 449:
        fault(f"plan has {len(units)} units")
    counts, last_sel, histories = {}, {}, {}
    for u in units:
        udir = root / "units" / u["unit_id"]
        res = json.loads((udir / "result.json").read_text(encoding="utf-8"))
        hbytes = (udir / res["history_file"]).read_bytes()
        if hashlib.sha256(hbytes).hexdigest().upper() != str(res["history_sha256"]).upper():
            fault(f"history sha256 differs from result.json for {u['unit_id']}")
        rows = list(csv.DictReader(io.StringIO(hbytes.decode("utf-8"))))
        if len(rows) != int(u["epochs"]):
            fault(f"{u['unit_id']}: {len(rows)} rows for {u['epochs']} epochs")
        best = min(range(len(rows)), key=lambda i: (float(rows[i]["val_loss"]), i))
        acc = float(rows[best]["test_acc"])
        c = round(acc * N_TEST)
        if abs(acc * N_TEST - c) > 1e-6:
            fault(f"non-integer count {u['unit_id']} {acc}")
        key = (u["condition"], u["model"], int(u["seed"]))
        if u["family"] == "MC003_anchor":
            histories[u["unit_id"]] = hbytes
            continue
        if key in counts:
            fault(f"duplicate cell and seed {key}")
        counts[key] = c
        last_sel[key] = (best + 1) == len(rows)
    if len(counts) != 440 or len(histories) != 9:
        fault(f"{len(counts)} evidence units and {len(histories)} anchors")

    # contrasts
    res = {}
    for c in contrast_defs():
        d = [counts[(c["cond"], c["A"], s)] - counts[(c["cond"], c["B"], s)] for s in SEEDS]
        nz = [x for x in d if x != 0]
        if not nz:
            p, method = 1.0, "all_zero"
        elif len(nz) == len(d) and len({abs(x) for x in d}) == len(d) and len(d) <= 50:
            p, method = exact_p(d), "exact"
        else:
            p, method = approx_p(d), "approx"
        lo, hi = boot_ci(d, c["boot_seed"])
        sd = statistics.stdev(d)
        res[c["id"]] = dict(c, d=d, p=p, method=method, mean_pp=100.0 * statistics.mean(d) / N_TEST, lo=lo, hi=hi,
                            lo_pp=100.0 * lo / N_TEST, hi_pp=100.0 * hi / N_TEST,
                            d_z=None if sd == 0 else statistics.mean(d) / sd)
    for g in ("D1", "D2", "D1_e100", "D2_e100"):
        ids = [k for k, v in res.items() if v["group"] == g]
        if len(ids) != (4 if g.startswith("D1") else 3):
            fault(f"family {g} has {len(ids)} contrasts")
        for k, ph in zip(ids, holm([res[i]["p"] for i in ids])):
            r = res[k]
            r["p_holm"] = ph
            r["label"] = ("no_difference" if r["method"] == "all_zero" else "supported" if ph < 0.05 and r["lo"] > 0
                          else "reverse" if ph < 0.05 and r["hi"] < 0 else "not_supported")
    comp = {}
    for g, name in (("D1", "D1_transfer"), ("D1_e100", "D1_transfer_e100")):
        sup = sum(1 for v in res.values() if v["group"] == g and v.get("transfer") and v["label"] == "supported")
        comp[name] = "transfer_observed" if sup == 2 else "dataset_specific" if sup == 1 else "not_observed"
    comp["D2_increment"] = res["D2:cifar100_full:dann_lrf-naive_branch"]["label"]
    comp["D2_increment_e100"] = res["D2_e100:cifar100_full_e100:dann_lrf-naive_branch"]["label"]
    differing = [k for k, v in res.items() if v["group"].endswith("_e100") and v["label"] != res[k.replace("_e100", "")]["label"]]
    comp["budget_agreement"] = "budget_dependent" if differing else "consistent"

    # cells
    cells = {}
    for cond, model, seed in cell_defs():
        c = [counts[(cond, model, s)] for s in SEEDS]
        lo, hi = boot_ci(c, seed)
        cells[(cond, model)] = {"mean_pp": 100.0 * statistics.mean(c) / N_TEST, "sd_pp": 100.0 * statistics.stdev(c) / N_TEST,
                                "lo_pp": 100.0 * lo / N_TEST, "hi_pp": 100.0 * hi / N_TEST,
                                "last": sum(1 for s in SEEDS if last_sel[(cond, model, s)])}

    # anchors against the frozen archive
    anchor_equal = {}
    with tarfile.open(frozen_arc, "r:gz") as tar:
        names = {m.name: m for m in tar.getmembers() if m.isfile()}
        for uid, hb in histories.items():
            hist = [n for n in names if f"/units/{uid}/" in n and n.endswith("history.csv")]
            if len(hist) != 1:
                anchor_equal[uid] = False
                continue
            anchor_equal[uid] = tar.extractfile(names[hist[0]]).read() == hb

    # comparison with the analysis outputs
    def rows(name):
        with open(proc / name, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    con = {r["id"]: r for r in rows("contrasts.csv")}
    dec = {r["criterion_id"]: r for r in rows("decision_inputs.csv")}
    summ = {(r["condition"], r["model"]): r for r in rows("summary_by_cell.csv")}
    cmp_rows = {r["rule_id"]: r for r in rows("composite_decisions.csv")}
    anc = {r["anchor_unit"]: r for r in rows("anchors.csv")}
    if set(con) != set(res) or set(dec) != set(res) or set(summ) != set(cells):
        fault("id sets differ between the recomputation and the analysis outputs")

    def close(a, b, tol=1e-9):
        return abs(float(a) - float(b)) <= tol + 1e-6 * abs(float(b))

    disagreements = []
    for k, r in res.items():
        c, dd = con[k], dec[k]
        checks = {"mean": close(dd["mean_diff_pp"], r["mean_pp"]), "ci": close(dd["ci_low_pp"], r["lo_pp"]) and close(dd["ci_high_pp"], r["hi_pp"]),
                  "method": c["wilcoxon_method"] == r["method"], "p": close(c["p_two_sided"], r["p"]), "p_holm": close(dd["holm_p"], r["p_holm"]),
                  "label": dd["outcome"] == r["label"], "differences": c["differences_count"] == ";".join(map(str, r["d"]))}
        if not all(checks.values()):
            disagreements.append({k: [n for n, ok in checks.items() if not ok]})
    for key, v in cells.items():
        s = summ[key]
        ok = (close(s["acc_mean_pp"], v["mean_pp"]) and close(s["acc_sd_pp"], v["sd_pp"]) and close(s["acc_ci95_low_pp"], v["lo_pp"])
              and close(s["acc_ci95_high_pp"], v["hi_pp"]) and int(s["selected_last_epoch_count"]) == v["last"])
        if not ok:
            disagreements.append({f"cell {key}": "values differ"})
    for name, outcome in comp.items():
        if cmp_rows.get(name, {}).get("outcome") != outcome:
            disagreements.append({f"composite {name}": [cmp_rows.get(name, {}).get("outcome"), outcome]})
    for uid, eq in anchor_equal.items():
        if (anc.get(uid, {}).get("bytes_equal") == "True") != eq:
            disagreements.append({f"anchor {uid}": eq})
    labels = {}
    for v in res.values():
        labels[v["label"]] = labels.get(v["label"], 0) + 1
    if sum(labels.values()) != 14:
        fault("label tally")
    doc = {"schema_version": 1, "method": "independent re-derivation from raw histories; exact Wilcoxon by DP; own Holm, labels, "
                                          "bootstrap, cells, composites and anchor comparison; no code shared with the analysis",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(), "units": len(counts) + len(histories),
           "contrasts": len(res), "cells": len(cells), "anchors_byte_equal": sum(anchor_equal.values()), "composites": comp,
           "label_tally": labels, "differing_between_budgets": differing, "disagreements": disagreements,
           "verdict": "PASS" if not disagreements else "DISAGREE",
           "rows": {k: {"label": v["label"], "method": v["method"], "mean_pp": v["mean_pp"], "ci95_pp": [v["lo_pp"], v["hi_pp"]],
                        "p": v["p"], "p_holm": v["p_holm"], "d_z": v["d_z"]} for k, v in res.items()}}
    out_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"units {doc['units']}; contrasts {len(res)}; cells {len(cells)}; anchors byte-equal {doc['anchors_byte_equal']}/9; "
          f"composites {comp}; disagreements {len(disagreements)}")
    for x in disagreements:
        print("DISAGREE", x)
    return 0 if not disagreements else 2


if __name__ == "__main__":
    sys.exit(main())
