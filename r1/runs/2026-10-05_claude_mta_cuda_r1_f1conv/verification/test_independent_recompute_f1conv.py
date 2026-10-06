"""Test of independent_recompute_f1conv.py on synthetic data (MC-NEURO-R1-004). Not evidence.

    python test_independent_recompute_f1conv.py <dann_benchmark root>

Positive control: the analysis of a synthetic arm and the recomputation agree (exit 0, verdict PASS, frozen labels rebuilt).
Negative controls: a flipped label, a shifted cell mean and a flipped prefix flag in the analysis outputs must give exit 2; a plan
with a missing unit must give a TOOL FAULT (exit 3). Operation neucom-r1-f1conv-protocol-launch-20261005 (Cowork-Claude,
claude-opus-5-5, max).
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DB = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(DB))
from r1 import aggregate_f1conv as af  # noqa: E402  (test harness only; the recompute script imports nothing from r1/)

SCRIPT = Path(__file__).resolve().parent / "independent_recompute_f1conv.py"


def run_recompute(s: dict, proc: Path, out: Path, plan: Path | None = None) -> int:
    args = [sys.executable, "-B", str(SCRIPT), str(s["root"]), str(plan or s["plan"]), str(proc), str(s["frozen_archive"]),
            str(s["frozen_contrasts"]), str(out)]
    r = subprocess.run(args, capture_output=True)
    print("   recompute:", r.stdout.decode("utf-8", "replace").strip().splitlines()[-1:] or r.stderr.decode()[-300:])
    return r.returncode


def edit_csv(path: Path, key: str, keyval, col: str, fn) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    hit = 0
    for r in rows:
        if r[key] == keyval:
            r[col] = fn(r[col])
            hit += 1
    assert hit == 1, (path.name, keyval, hit)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        s = af._synthetic(tmp)
        proc = tmp / "proc"
        args = af._args(s, proc)
        with contextlib.redirect_stdout(io.StringIO()):
            rc_a = af.analyse(args)
        results.append(("synthetic analysis exits 0", rc_a == 0))
        rc = run_recompute(s, proc, tmp / "rec.json")
        doc = json.loads((tmp / "rec.json").read_text(encoding="utf-8"))
        results.append(("positive control: recompute agrees (exit 0, PASS)", rc == 0 and doc["verdict"] == "PASS"))
        results.append(("positive control: frozen labels rebuilt equal the frozen file", doc["frozen_labels_rebuilt_equal_file"] is True))
        results.append(("positive control: 240 prefixes byte-equal", doc["prefix_byte_equal"] == 240))

        def negative(name, path_name, key, keyval, col, fn):
            p2 = tmp / f"proc_{name}"
            shutil.copytree(proc, p2)
            edit_csv(p2 / path_name, key, keyval, col, fn)
            rc_n = run_recompute(s, p2, tmp / f"rec_{name}.json")
            results.append((f"negative control {name}: exit 2", rc_n == 2))

        flip = {"supported": "not_supported", "not_supported": "supported", "reverse": "not_supported", "no_difference": "supported"}
        negative("flipped label", "decision_inputs.csv", "criterion_id", "F1e100:kmnist_full_e100:dann_lrf-naive_branch", "outcome",
                 lambda v: flip[v])
        p3 = tmp / "proc_cell"
        shutil.copytree(proc, p3)
        rows = list(csv.DictReader(open(p3 / "summary_by_cell.csv", encoding="utf-8", newline="")))
        rows[4]["acc_mean_pp"] = repr(float(rows[4]["acc_mean_pp"]) + 0.01)
        with open(p3 / "summary_by_cell.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        results.append(("negative control shifted cell mean: exit 2", run_recompute(s, p3, tmp / "rec_cell.json") == 2))
        negative("flipped prefix flag", "prefix_gate.csv", "unit_id", "cifar_full_e100__dann_random__s13", "prefix_bytes_equal",
                 lambda v: "False" if v == "True" else "True")
        plan = json.loads(Path(s["plan"]).read_text(encoding="utf-8"))
        plan["units"] = plan["units"][:-1]
        bad_plan = tmp / "plan_239.json"
        bad_plan.write_text(json.dumps(plan), encoding="utf-8")
        results.append(("negative control plan with 239 units: TOOL FAULT exit 3", run_recompute(s, proc, tmp / "rec_plan.json", bad_plan) == 3))
    failed = [n for n, ok in results if not ok]
    for n, ok in results:
        print("PASS" if ok else "FAIL", n)
    print(f"test_independent_recompute_f1conv: {len(results)} checks, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
