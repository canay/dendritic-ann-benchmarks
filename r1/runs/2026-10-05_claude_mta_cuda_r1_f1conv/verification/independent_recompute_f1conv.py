"""Independent recomputation of the F1 convergence-arm decision inputs (MC-NEURO-R1-004, protocol Section 11, A18).

Second method: shares no code with r1/aggregate_f1conv.py or r1/aggregate_r1.py (nothing is imported from r1/). The contrast,
cell and seed definitions below are written from the protocol text, not copied from the analysis script.

    python independent_recompute_f1conv.py <run root holding units/> <plan_f1conv_run.json> <processed_outputs dir>
        <frozen outputs archive.tar.gz> <frozen contrasts.csv> <out.json>

1. Every planned unit's correct count is re-derived from its own history.csv (first epoch of minimum val_loss; A9) and must be a
   whole number of the 10,000 test images; the history must hold exactly 100 epochs. The frozen 30-epoch counterpart of every
   unit (same condition without the suffix _e100, same model and seed) is read from the frozen archive the same way.
2. Prefix gate: the header and the first 30 data lines of every 100-epoch history equal the frozen history byte for byte.
3. The nine contrasts (per dataset: DANN-LRF minus MLP-Param, minus Naive-Branch, minus DANN-RANDOM) are rebuilt from the counts:
   mean difference, two-sided Wilcoxon signed-rank p (EXACT by dynamic programming when no difference is zero, no absolute
   difference is tied and n <= 50; otherwise the normal approximation with average ranks, tie-corrected variance and no continuity
   correction), Holm within each dataset, the 95 % percentile bootstrap CI with R = 10,000 and seed 20261005001 + k, d_z, and the
   A8 label. The frozen 30-epoch labels are rebuilt the same way from the frozen counts with the frozen seeds 20261003001-009 and
   compared with the frozen contrasts file.
4. Budget agreement per contrast; the two-of-three summary at 100 epochs; the per-seed budget change of each paired difference
   (mean, bootstrap CI with seed 20261005201 + k).
5. The twelve cells: mean accuracy, SD, bootstrap CI with seed 20261005101 + k, the number of seeds selected at the last epoch,
   the accuracy gain over the frozen cell.
Every value is compared with the analysis outputs. Internal invariants are asserted; a failed invariant prints TOOL FAULT and exits
3. Exit 0 when everything agrees, 2 when anything disagrees.
Written before any result of the arm existed. Operation neucom-r1-f1conv-protocol-launch-20261005 (Cowork-Claude,
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
DATASETS = [("fashion_full", "fashionmnist"), ("kmnist_full", "kmnist"), ("cifar_full", "cifar10")]
MODELS = ["dann_lrf", "naive_branch", "mlp_param", "dann_random"]
COMPARATORS = ["mlp_param", "naive_branch", "dann_random"]  # the order of the frozen manifest


def fault(msg: str) -> None:
    print("TOOL FAULT:", msg)
    sys.exit(3)


def contrast_defs():
    out = []
    for base, _ds in DATASETS:
        for b in COMPARATORS:
            out.append({"id": f"F1e100:{base}_e100:dann_lrf-{b}", "frozen_id": f"F1:{base}:dann_lrf-{b}", "group": base,
                        "cond": base + "_e100", "base": base, "A": "dann_lrf", "B": b, "rq2": b == "naive_branch"})
    for k, c in enumerate(out):
        c["boot_seed"] = 20261005001 + k
        c["change_seed"] = 20261005201 + k
        c["frozen_seed"] = 20261003001 + k  # frozen manifest: fashion 001-003, kmnist 004-006, cifar 007-009 in this order
    return out


def cell_defs():
    out = [(base + "_e100", base, m) for base, _ds in DATASETS for m in MODELS]
    return [(c, b, m, 20261005101 + k) for k, (c, b, m) in enumerate(out)]


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


def wilcoxon_p(d):
    nz = [x for x in d if x != 0]
    if not nz:
        return 1.0, "all_zero"
    if len(nz) == len(d) and len({abs(x) for x in d}) == len(d) and len(d) <= 50:
        return exact_p(d), "exact"
    return approx_p(d), "approx"


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


def a8(method, p_holm, lo, hi):
    if method == "all_zero":
        return "no_difference"
    if p_holm < 0.05 and lo > 0:
        return "supported"
    if p_holm < 0.05 and hi < 0:
        return "reverse"
    return "not_supported"


def selected(hbytes: bytes, epochs: int, uid: str):
    rows = list(csv.DictReader(io.StringIO(hbytes.decode("utf-8"))))
    if len(rows) != epochs:
        fault(f"{uid}: {len(rows)} rows for {epochs} epochs")
    best = min(range(len(rows)), key=lambda i: (float(rows[i]["val_loss"]), i))
    acc = float(rows[best]["test_acc"])
    c = round(acc * N_TEST)
    if abs(acc * N_TEST - c) > 1e-6:
        fault(f"non-integer count {uid} {acc}")
    return c, best + 1, rows


def main() -> int:
    root, plan_path, proc, frozen_arc, frozen_con, out_path = (Path(a) for a in sys.argv[1:7])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    units = plan["units"]
    if len(units) != 240 or len({u["unit_id"] for u in units}) != 240:
        fault(f"plan has {len(units)} units")
    if {u["family"] for u in units} != {"MC004_F1_e100"} or {u["epochs"] for u in units} != {100}:
        fault("plan family or epochs differ from A18")
    counts, sel_epoch, hist = {}, {}, {}
    for u in units:
        udir = root / "units" / u["unit_id"]
        res = json.loads((udir / "result.json").read_text(encoding="utf-8"))
        hb = (udir / res["history_file"]).read_bytes()
        if hashlib.sha256(hb).hexdigest().upper() != str(res["history_sha256"]).upper():
            fault(f"history sha256 differs from result.json for {u['unit_id']}")
        c, e, _rows = selected(hb, 100, u["unit_id"])
        key = (u["condition"], u["model"], int(u["seed"]))
        if key in counts:
            fault(f"duplicate cell and seed {key}")
        counts[key], sel_epoch[key], hist[key] = c, e, hb
    if len(counts) != 240:
        fault(f"{len(counts)} units")

    # frozen counterparts from the frozen archive
    fcounts, fhist = {}, {}
    with tarfile.open(frozen_arc, "r:gz") as tar:
        names = {m.name: m for m in tar.getmembers() if m.isfile()}
        for (cond, model, seed) in counts:
            base = cond[: -len("_e100")]
            uid = f"{base}__{model}__s{seed:02d}"
            h = [n for n in names if f"/units/{uid}/" in n and n.endswith("history.csv")]
            if len(h) != 1:
                fault(f"frozen history of {uid}: {len(h)} members")
            fb = tar.extractfile(names[h[0]]).read()
            fc, _fe, _fr = selected(fb, 30, uid)
            fcounts[(base, model, seed)], fhist[(base, model, seed)] = fc, fb

    # prefix gate
    prefix = {}
    for (cond, model, seed), hb in hist.items():
        fb = fhist[(cond[: -len("_e100")], model, seed)]
        prefix[(cond, model, seed)] = hb.split(b"\n")[:31] == fb.split(b"\n")[:31]

    # contrasts at 100 epochs and the frozen labels rebuilt
    res, frozen_labels = {}, {}
    for c in contrast_defs():
        d = [counts[(c["cond"], c["A"], s)] - counts[(c["cond"], c["B"], s)] for s in SEEDS]
        d30 = [fcounts[(c["base"], c["A"], s)] - fcounts[(c["base"], c["B"], s)] for s in SEEDS]
        p, method = wilcoxon_p(d)
        lo, hi = boot_ci(d, c["boot_seed"])
        sd = statistics.stdev(d)
        ch = [x - y for x, y in zip(d, d30)]
        clo, chi = boot_ci(ch, c["change_seed"])
        p30, m30 = wilcoxon_p(d30)
        lo30, hi30 = boot_ci(d30, c["frozen_seed"])
        res[c["id"]] = dict(c, d=d, d30=d30, p=p, method=method, lo=lo, hi=hi, mean_pp=100.0 * statistics.mean(d) / N_TEST,
                            lo_pp=100.0 * lo / N_TEST, hi_pp=100.0 * hi / N_TEST, d_z=None if sd == 0 else statistics.mean(d) / sd,
                            change_pp=100.0 * statistics.mean(ch) / N_TEST, change_lo_pp=100.0 * clo / N_TEST,
                            change_hi_pp=100.0 * chi / N_TEST, p30=p30, m30=m30, lo30=lo30, hi30=hi30)
    for g in ("fashion_full", "kmnist_full", "cifar_full"):
        ids = [k for k, v in res.items() if v["group"] == g]
        if len(ids) != 3:
            fault(f"family {g} has {len(ids)} contrasts")
        for k, ph in zip(ids, holm([res[i]["p"] for i in ids])):
            res[k]["p_holm"] = ph
            res[k]["label"] = a8(res[k]["method"], ph, res[k]["lo"], res[k]["hi"])
        for k, ph in zip(ids, holm([res[i]["p30"] for i in ids])):
            frozen_labels[res[k]["frozen_id"]] = a8(res[k]["m30"], ph, res[k]["lo30"], res[k]["hi30"])
    with open(frozen_con, encoding="utf-8", newline="") as f:
        frozen_file = {r["id"]: r["outcome"] for r in csv.DictReader(f)}
    frozen_rebuilt_ok = all(frozen_file.get(k) == v for k, v in frozen_labels.items())
    agreement = {k: ("consistent" if frozen_labels[v["frozen_id"]] == v["label"] else "budget_dependent") for k, v in res.items()}
    sup = sum(1 for v in res.values() if v["rq2"] and v["label"] == "supported")
    comp = {"RQ2_robustness_e100": "robust" if sup >= 2 else "not_robust",
            "budget_agreement": "budget_dependent" if any(a != "consistent" for a in agreement.values()) else "consistent",
            "prefix_gate": "pass" if all(prefix.values()) else "determinism_finding"}
    sup30 = sum(1 for v in res.values() if v["rq2"] and frozen_labels[v["frozen_id"]] == "supported")
    comp["RQ2_robustness_frozen30"] = "robust" if sup30 >= 2 else "not_robust"

    # cells
    cells = {}
    for cond, base, model, seed in cell_defs():
        c = [counts[(cond, model, s)] for s in SEEDS]
        fc = [fcounts[(base, model, s)] for s in SEEDS]
        lo, hi = boot_ci(c, seed)
        cells[(cond, model)] = {"mean_pp": 100.0 * statistics.mean(c) / N_TEST, "sd_pp": 100.0 * statistics.stdev(c) / N_TEST,
                                "lo_pp": 100.0 * lo / N_TEST, "hi_pp": 100.0 * hi / N_TEST,
                                "last": sum(1 for s in SEEDS if sel_epoch[(cond, model, s)] == 100),
                                "gain_pp": 100.0 * (statistics.mean(c) - statistics.mean(fc)) / N_TEST}

    # comparison with the analysis outputs
    def rows(name):
        with open(proc / name, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    con = {r["id"]: r for r in rows("contrasts.csv")}
    dec = {r["criterion_id"]: r for r in rows("decision_inputs.csv")}
    summ = {(r["condition"], r["model"]): r for r in rows("summary_by_cell.csv")}
    agr = {r["id"]: r for r in rows("budget_agreement.csv")}
    chg = {r["id"]: r for r in rows("budget_change.csv")}
    gate = {(r["condition"], r["model"], int(r["seed"])): r for r in rows("prefix_gate.csv")}
    cmp_rows = {r["rule_id"]: r for r in rows("composite_decisions.csv")}
    if set(con) != set(res) or set(dec) != set(res) or set(summ) != set(cells) or set(agr) != set(res) or set(gate) != set(prefix):
        fault("id sets differ between the recomputation and the analysis outputs")

    def close(a, b, tol=1e-9):
        return abs(float(a) - float(b)) <= tol + 1e-6 * abs(float(b))

    disagreements = []
    if not frozen_rebuilt_ok:
        disagreements.append({"frozen labels": "rebuilt frozen labels differ from the frozen contrasts file"})
    for k, r in res.items():
        c, dd, a, h = con[k], dec[k], agr[k], chg[k]
        checks = {"mean": close(dd["mean_diff_pp"], r["mean_pp"]), "ci": close(dd["ci_low_pp"], r["lo_pp"]) and close(dd["ci_high_pp"], r["hi_pp"]),
                  "method": c["wilcoxon_method"] == r["method"], "p": close(c["p_two_sided"], r["p"]), "p_holm": close(dd["holm_p"], r["p_holm"]),
                  "label": dd["outcome"] == r["label"], "differences": c["differences_count"] == ";".join(map(str, r["d"])),
                  "frozen_label": a["frozen30_outcome"] == frozen_labels[r["frozen_id"]], "agreement": a["agreement"] == agreement[k],
                  "change": close(h["budget_change_mean_pp"], r["change_pp"]) and close(h["budget_change_ci95_low_pp"], r["change_lo_pp"])
                  and close(h["budget_change_ci95_high_pp"], r["change_hi_pp"]),
                  "frozen_differences": h["frozen30_differences_count"] == ";".join(map(str, r["d30"]))}
        if not all(checks.values()):
            disagreements.append({k: [n for n, ok in checks.items() if not ok]})
    for key, v in cells.items():
        s = summ[key]
        ok = (close(s["acc_mean_pp"], v["mean_pp"]) and close(s["acc_sd_pp"], v["sd_pp"]) and close(s["acc_ci95_low_pp"], v["lo_pp"])
              and close(s["acc_ci95_high_pp"], v["hi_pp"]) and int(s["selected_last_epoch_count"]) == v["last"]
              and close(s["acc_gain_over_frozen30_pp"], v["gain_pp"]))
        if not ok:
            disagreements.append({f"cell {key}": "values differ"})
    for key, eq in prefix.items():
        if (gate[key]["prefix_bytes_equal"] == "True") != eq:
            disagreements.append({f"prefix {key}": eq})
    for name, outcome in comp.items():
        if cmp_rows.get(name, {}).get("outcome") != outcome:
            disagreements.append({f"composite {name}": [cmp_rows.get(name, {}).get("outcome"), outcome]})
    tally = {}
    for v in res.values():
        tally[v["label"]] = tally.get(v["label"], 0) + 1
    if sum(tally.values()) != 9:
        fault("label tally")
    doc = {"schema_version": 1, "method": "independent re-derivation from raw histories (arm and frozen archive); exact Wilcoxon by "
                                          "DP; own Holm, labels, bootstrap, cells, budget agreement and change, prefix comparison; no "
                                          "code shared with the analysis",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(), "units": len(counts),
           "contrasts": len(res), "cells": len(cells), "prefix_byte_equal": sum(prefix.values()), "composites": comp,
           "frozen_labels_rebuilt_equal_file": frozen_rebuilt_ok, "label_tally": tally,
           "differing_between_budgets": [k for k, a in agreement.items() if a != "consistent"], "disagreements": disagreements,
           "verdict": "PASS" if not disagreements else "DISAGREE",
           "rows": {k: {"label": v["label"], "frozen30_label": frozen_labels[v["frozen_id"]], "method": v["method"], "mean_pp": v["mean_pp"],
                        "ci95_pp": [v["lo_pp"], v["hi_pp"]], "p": v["p"], "p_holm": v["p_holm"], "d_z": v["d_z"],
                        "budget_change_pp": v["change_pp"]} for k, v in res.items()}}
    out_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"units {len(counts)}; contrasts {len(res)}; cells {len(cells)}; prefix byte-equal {doc['prefix_byte_equal']}/240; "
          f"frozen labels rebuilt {frozen_rebuilt_ok}; composites {comp}; disagreements {len(disagreements)}")
    for x in disagreements:
        print("DISAGREE", x)
    return 0 if not disagreements else 2


if __name__ == "__main__":
    sys.exit(main())
