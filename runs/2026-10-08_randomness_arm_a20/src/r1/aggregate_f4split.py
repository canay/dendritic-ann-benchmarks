"""Analysis of the MC-NEURO-R1-005 arm (protocol Section 12, amendment A21): F4 with the data source split into a split seed
and an order seed. Descriptive only (as F4): no test, no label, no variance shares, no ranking of sources.

    python -B r1/aggregate_f4split.py selftest
    python -B r1/aggregate_f4split.py analyse --plan <plan.json> --root <extracted run folder> --identity <IDENTITY.json>
        --frozen-receipt <frozen TRANSFER_RECEIPT_outputs.json> --out <processed_outputs dir>

Primary metric: test accuracy at the validation-selected epoch (first minimum of the validation loss), on the integer scale
of correct test images (the FashionMNIST test set has 10,000 images), re-derived from every history and checked against the
unit's summary by r1_runner.validate_result. Differences are DANN-LRF minus Naive-Branch in percentage points.

Per source (split: values 0-9 of MC005_F4_split; order: value 0 = the split sweep's value-0 unit, the shared background
point, and values 1-9 of MC005_F4_order) and per model: mean, sample SD, min and max of the accuracy over the ten values; for
the paired difference: mean, sample SD, min, max and the counts of positive, zero and negative differences.

Reproduction gate: the histories of the 8 anchor units and of the two value-0 split units must equal, byte for byte, the
frozen F4 data-seed units of the same model and value (frozen transfer receipt member sha256). The gate is reported, never
used to drop a unit.

Python standard library only (no NumPy/SciPy), so no pinned-version question arises. Fail-closed: a missing, invalid or
foreign unit stops the analysis with exit 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1 import r1_plan  # noqa: E402
from r1.r1_runner import atomic_write_json, sha256_file, unit_spec, unit_spec_sha256, validate_result  # noqa: E402

N_TEST = 10000
MODELS = ("dann_lrf", "naive_branch")
SOURCES = ("split", "order")
FROZEN_UNIT = "fashion_full_rand_data__{model}__s{v:02d}"


class AnalysisError(Exception):
    pass


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def correct_count(acc: float) -> int:
    c = round(acc * N_TEST)
    if abs(c - acc * N_TEST) > 1e-6:
        raise AnalysisError(f"accuracy {acc!r} is not a whole number of test images")
    return int(c)


def load_units(plan: dict, root: Path, identity: dict) -> dict:
    """unit_id -> {'unit', 'correct', 'history_sha256', 'best_val_epoch'}; every planned unit must validate."""
    out = {}
    for u in plan["units"]:
        udir = root / "units" / u["unit_id"]
        why = validate_result(u, udir, identity)
        if why is not None:
            raise AnalysisError(f"{u['unit_id']}: {why}")
        res = json.loads((udir / "result.json").read_text(encoding="utf-8"))
        hist = udir / res["history_file"]
        out[u["unit_id"]] = {"unit": u, "correct": correct_count(float(res["summary"]["test_acc_at_best_val"])),
                             "history_sha256": sha256_file(hist), "best_val_epoch": int(res["summary"]["best_val_epoch"])}
    extra = sorted(p.name for p in (root / "units").iterdir() if p.is_dir() and p.name not in out)
    if extra:
        raise AnalysisError(f"units outside the plan: {extra[:5]}")
    return out


def source_cells(units: dict) -> dict:
    """(source, model) -> [(value, correct)] for values 0-9."""
    cells = {}
    for rec in units.values():
        u = rec["unit"]
        ex = u["extra"]
        if u["family"] == "MC005_F4_split":
            cells.setdefault(("split", u["model"]), []).append((ex["split_seed"], rec["correct"]))
            if ex["split_seed"] == 0:
                cells.setdefault(("order", u["model"]), []).append((0, rec["correct"]))
        elif u["family"] == "MC005_F4_order":
            cells.setdefault(("order", u["model"]), []).append((ex["order_seed"], rec["correct"]))
    for key, vals in cells.items():
        vals.sort()
        if [v for v, _ in vals] != list(range(10)):
            raise AnalysisError(f"cell {key} does not hold values 0-9: {[v for v, _ in vals]}")
    if sorted(cells) != sorted((s, m) for s in SOURCES for m in MODELS):
        raise AnalysisError(f"cells {sorted(cells)}")
    return cells


def pp(count) -> float:
    return float(Fraction(count) * 100 / N_TEST)


def describe(values_pp: list) -> dict:
    return {"n": len(values_pp), "mean": statistics.fmean(values_pp), "sd": statistics.stdev(values_pp),
            "min": min(values_pp), "max": max(values_pp)}


def summarise(cells: dict) -> tuple:
    model_rows, paired_rows = [], []
    for s in SOURCES:
        for m in MODELS:
            d = describe([pp(c) for _, c in cells[(s, m)]])
            model_rows.append({"source": s, "model": m, **{f"acc_{k}": v for k, v in d.items()}})
        a = dict(cells[(s, "dann_lrf")])
        b = dict(cells[(s, "naive_branch")])
        diffs = [a[v] - b[v] for v in range(10)]
        d = describe([pp(x) for x in diffs])
        paired_rows.append({"source": s, **{f"diff_{k}": v for k, v in d.items()},
                            "n_positive": sum(x > 0 for x in diffs), "n_zero": sum(x == 0 for x in diffs),
                            "n_negative": sum(x < 0 for x in diffs),
                            "diffs_correct_images": ";".join(str(x) for x in diffs)})
    return model_rows, paired_rows


def gate(units: dict, receipt: dict) -> list:
    rows = []
    members = receipt["members"]
    for rec in units.values():
        u = rec["unit"]
        ex = u["extra"]
        if u["family"] == "MC005_anchor":
            v = u["seed"]
        elif u["family"] == "MC005_F4_split" and ex["split_seed"] == 0:
            v = 0
        else:
            continue
        name = FROZEN_UNIT.format(model=u["model"], v=v)
        hits = [k for k in members if f"/units/{name}/" in k and k.endswith("/history.csv")]
        if len(hits) != 1:
            raise AnalysisError(f"frozen receipt has {len(hits)} history files for {name}")
        want = members[hits[0]]["sha256"].upper()
        rows.append({"unit_id": u["unit_id"], "frozen_unit": name, "history_sha256": rec["history_sha256"],
                     "frozen_history_sha256": want, "equal": rec["history_sha256"] == want})
    rows.sort(key=lambda r: r["unit_id"])
    if len(rows) != 10:
        raise AnalysisError(f"gate rows {len(rows)} != 10")
    return rows


def write_csv(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def analyse(plan_p: Path, root: Path, ident_p: Path, receipt_p: Path, out: Path) -> dict:
    plan = json.loads(plan_p.read_text(encoding="utf-8"))
    identity = json.loads(ident_p.read_text(encoding="utf-8"))
    if plan["units"] != r1_plan.f4split_run():
        raise AnalysisError("the plan is not the f4split_run plan of this builder")
    if identity.get("plan_sha256", "").upper() != sha(plan_p):
        raise AnalysisError("identity plan_sha256 differs from the plan file")
    units = load_units(plan, root, identity)
    cells = source_cells(units)
    model_rows, paired_rows = summarise(cells)
    gate_rows = gate(units, json.loads(receipt_p.read_text(encoding="utf-8")))
    unit_rows = [{"unit_id": k, "family": r["unit"]["family"], "model": r["unit"]["model"], "seed": r["unit"]["seed"],
                  "split_seed": r["unit"]["extra"].get("split_seed", ""), "order_seed": r["unit"]["extra"].get("order_seed", ""),
                  "data_seed": r["unit"]["extra"].get("data_seed", ""), "correct": r["correct"],
                  "acc_pp": pp(r["correct"]), "best_val_epoch": r["best_val_epoch"], "history_sha256": r["history_sha256"]}
                 for k, r in sorted(units.items())]
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise AnalysisError(f"output folder {out} is not empty")
    write_csv(out / "unit_table.csv", unit_rows)
    write_csv(out / "source_model_summary.csv", model_rows)
    write_csv(out / "source_paired_summary.csv", paired_rows)
    write_csv(out / "reproduction_gate.csv", gate_rows)
    decision = {
        "criteria": [
            {"criterion_id": f"{r['source']}_positive_count", "value": r["n_positive"], "of": r["diff_n"],
             "all_positive": r["n_positive"] == r["diff_n"], "mean_diff_pp": r["diff_mean"], "sd_diff_pp": r["diff_sd"],
             "min_diff_pp": r["diff_min"], "direction_expected": "positive",
             "direction_observed": "positive" if r["diff_mean"] > 0 else ("negative" if r["diff_mean"] < 0 else "zero")}
            for r in paired_rows],
        "reproduction_gate": {"rows": len(gate_rows), "equal": sum(g["equal"] for g in gate_rows),
                              "pass": all(g["equal"] for g in gate_rows)},
        "discriminators": "per-source count of positive paired differences and its minimum; the gate pass flag",
    }
    atomic_write_json(out / "decision_inputs.json", decision)
    run = {"written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script": "dann_benchmark/r1/aggregate_f4split.py", "script_sha256": sha(Path(__file__)),
           "python": sys.version.split()[0], "inputs": {"plan": sha(plan_p), "identity": sha(ident_p),
                                                        "frozen_receipt": sha(receipt_p)},
           "units": len(units), "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    atomic_write_json(out / "ANALYSIS_RUN.json", run)
    return {"paired": paired_rows, "gate": decision["reproduction_gate"]}


# ---------------------------------------------------------------- self-test on synthetic outputs (never evidence)
def _synthetic_history(u: dict, rng: random.Random) -> list:
    rows = []
    for e in range(1, int(u["epochs"]) + 1):
        rows.append([u["dataset"], u["model"], u["seed"], e, 1.0 / e, 0.5 + e / 100, 1.0 / e + rng.random() / 10,
                     0.5 + e / 100, 1.0 / e, (8000 + rng.randrange(0, 1500)) / N_TEST])
    return rows


def _write_unit(root: Path, u: dict, rows: list, identity: dict) -> str:
    udir = root / "units" / u["unit_id"]
    (udir / "attempt_01").mkdir(parents=True)
    hist = udir / "attempt_01" / "history.csv"
    with open(hist, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model_name", "seed", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss",
                    "test_acc"])
        for r in rows:
            w.writerow(r)
    best = min(rows, key=lambda r: float(r[6]))
    res = {"schema_version": 1, "status": "completed", "unit_id": u["unit_id"], "unit_spec": unit_spec(u),
           "unit_spec_sha256": unit_spec_sha256(u), "attempt": 1, "history_file": "attempt_01/history.csv",
           "history_sha256": sha256_file(hist), "identity": identity,
           "summary": {"best_val_epoch": int(best[3]), "best_val_loss": float(best[6]), "test_acc_at_best_val": float(best[9]),
                       "test_loss_at_best_val": float(best[8]), "best_test_acc": max(float(r[9]) for r in rows),
                       "final_test_acc": float(rows[-1][9])}}
    atomic_write_json(udir / "result.json", res)
    return sha256_file(hist)


def selftest() -> int:
    checks, failed = 0, []

    def check(name, ok):
        nonlocal checks
        checks += 1
        if not ok:
            failed.append(name)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        units = r1_plan.f4split_run()
        plan = {"schema_version": 1, "run_id": "selftest", "family": "f4split_run", "units": units}
        plan_p = td / "plan.json"
        plan_p.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        identity = {"run_id": "selftest", "plan_sha256": sha(plan_p), "code_fingerprint": "X", "environment": {}}
        ident_p = td / "IDENTITY.json"
        ident_p.write_text(json.dumps(identity) + "\n", encoding="utf-8")
        root = td / "run"
        rng = random.Random(20261008)
        frozen_hashes, gate_hist = {}, {}
        # anchors and split value 0 get the same synthetic history as their frozen counterpart (keyed by model, value)
        for u in units:
            ex = u["extra"]
            key = None
            if u["family"] == "MC005_anchor":
                key = (u["model"], u["seed"])
            elif u["family"] == "MC005_F4_split" and ex["split_seed"] == 0:
                key = (u["model"], 0)
            if key is not None and key in gate_hist:
                rows = gate_hist[key]
            else:
                rows = _synthetic_history(u, rng)
                if key is not None:
                    gate_hist[key] = rows
            h = _write_unit(root, u, rows, identity)
            if key is not None:
                frozen_hashes[FROZEN_UNIT.format(model=key[0], v=key[1])] = h
        receipt = {"members": {f"frozen/units/{n}/attempt_01/history.csv": {"sha256": h} for n, h in frozen_hashes.items()}}
        rec_p = td / "receipt.json"
        rec_p.write_text(json.dumps(receipt), encoding="utf-8")
        res = analyse(plan_p, root, ident_p, rec_p, td / "out")
        check("gate passes on equal histories", res["gate"]["pass"] and res["gate"]["equal"] == 10)
        # hand computation of the split paired mean from the unit table
        ut = list(csv.DictReader(open(td / "out" / "unit_table.csv", encoding="utf-8")))
        check("unit table rows 46", len(ut) == 46)
        by = {(r["family"], r["model"], r["split_seed"], r["order_seed"]): int(r["correct"]) for r in ut}
        diffs = [by[("MC005_F4_split", "dann_lrf", str(v), "0")] - by[("MC005_F4_split", "naive_branch", str(v), "0")]
                 for v in range(10)]
        hand = sum(diffs) / 10 / 100
        got = [r for r in res["paired"] if r["source"] == "split"][0]
        check("split paired mean equals the hand computation", abs(got["diff_mean"] - hand) < 1e-12)
        check("split positive count equals the hand count", got["n_positive"] == sum(d > 0 for d in diffs))
        odiffs = [by[("MC005_F4_split", "dann_lrf", "0", "0")] - by[("MC005_F4_split", "naive_branch", "0", "0")]]
        odiffs += [by[("MC005_F4_order", "dann_lrf", "0", str(v))] - by[("MC005_F4_order", "naive_branch", "0", str(v))]
                   for v in range(1, 10)]
        ogot = [r for r in res["paired"] if r["source"] == "order"][0]
        check("order cell uses the shared value-0 unit", abs(ogot["diff_mean"] - sum(odiffs) / 1000) < 1e-12)
        check("decision inputs written", (td / "out" / "decision_inputs.json").is_file())
        # fail-closed paths
        out_again = td / "out"
        try:
            analyse(plan_p, root, ident_p, rec_p, out_again)
            check("non-empty output folder refused", False)
        except AnalysisError:
            check("non-empty output folder refused", True)
        bad_root = td / "bad"
        shutil.copytree(root, bad_root)
        victim = bad_root / "units" / units[20]["unit_id"] / "attempt_01" / "history.csv"
        victim.write_text(victim.read_text(encoding="utf-8").replace("0.5", "0.6", 1), encoding="utf-8")
        try:
            analyse(plan_p, bad_root, ident_p, rec_p, td / "out_bad1")
            check("tampered history refused", False)
        except AnalysisError:
            check("tampered history refused", True)
        miss_root = td / "miss"
        shutil.copytree(root, miss_root)
        shutil.rmtree(miss_root / "units" / units[30]["unit_id"])
        try:
            analyse(plan_p, miss_root, ident_p, rec_p, td / "out_bad2")
            check("missing unit refused", False)
        except AnalysisError:
            check("missing unit refused", True)
        foreign = dict(identity, code_fingerprint="Y")
        f_p = td / "IDENTITY_foreign.json"
        f_p.write_text(json.dumps(foreign) + "\n", encoding="utf-8")
        try:
            analyse(plan_p, root, f_p, rec_p, td / "out_bad3")
            check("foreign identity refused", False)
        except AnalysisError:
            check("foreign identity refused", True)
        receipt_bad = json.loads(rec_p.read_text(encoding="utf-8"))
        k0 = sorted(receipt_bad["members"])[0]
        receipt_bad["members"][k0]["sha256"] = "0" * 64
        rb_p = td / "receipt_bad.json"
        rb_p.write_text(json.dumps(receipt_bad), encoding="utf-8")
        res_b = analyse(plan_p, root, ident_p, rb_p, td / "out_gate_fail")
        planted = k0.split("/units/")[1].split("/")[0]
        hit = [r for r in csv.DictReader(open(td / "out_gate_fail" / "reproduction_gate.csv", encoding="utf-8"))
               if r["frozen_unit"] == planted]
        # the expected count is derived from the mapping, never assumed: every gate row mapped to the planted frozen unit fails
        check("gate reports a planted mismatch without crashing",
              len(hit) >= 1 and not res_b["gate"]["pass"] and res_b["gate"]["equal"] == 10 - len(hit)
              and all(r["equal"] == "False" for r in hit))
        same = all(sha(td / "out" / n) == sha(td / "out_gate_fail" / n)
                   for n in ("unit_table.csv", "source_model_summary.csv", "source_paired_summary.csv"))
        check("deterministic outputs for the same units", same)
    print(f"selftest: {checks} checks, failed {failed if failed else 'none'}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    a = sub.add_parser("analyse")
    a.add_argument("--plan", required=True)
    a.add_argument("--root", required=True)
    a.add_argument("--identity", required=True)
    a.add_argument("--frozen-receipt", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    try:
        res = analyse(Path(args.plan), Path(args.root), Path(args.identity), Path(args.frozen_receipt), Path(args.out))
    except AnalysisError as exc:
        print("ANALYSIS FAILED (fail-closed):", exc)
        return 2
    print(json.dumps(res["gate"]), "paired rows", len(res["paired"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
