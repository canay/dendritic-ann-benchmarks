"""Synthetic test of independent_recompute_tierd.py before any real result exists (MC-NEURO-R1-003).

Builds the synthetic tier D run of r1/aggregate_tierd.py's self-test (449 units, a synthetic frozen archive), runs the
analysis, then runs the independent script as a separate process: it must agree (exit 0, verdict PASS). Negative controls:
a tampered Holm p, a tampered cell mean and a tampered anchor flag must each make it disagree (exit 2).

    python -B test_independent_recompute_tierd.py <dann_benchmark dir>
Operation neucom-r1-tierd-protocol-lock-20261005 (Cowork-Claude, claude-opus-5-5, max).
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

bench = Path(sys.argv[1])
sys.path.insert(0, str(bench))
from r1 import aggregate_tierd as at  # noqa: E402  (test harness only; the recompute script imports nothing from r1/)

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "independent_recompute_tierd.py"
fails = []


def run_indep(tmp: Path, out_dir: Path, tag: str) -> tuple:
    out = tmp / f"indep_{tag}.json"
    p = subprocess.run([sys.executable, "-B", str(SCRIPT), str(tmp / "run"), str(tmp / "plan.json"), str(out_dir), str(tmp / "frozen.tar.gz"),
                        str(out)], capture_output=True, text=True)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    return p.returncode, doc, p.stdout + p.stderr


def tamper(src: Path, dst: Path, name: str, key_col: str, key: str, col: str, value: str) -> None:
    shutil.copytree(src, dst)
    with open(dst / name, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    hit = 0
    for r in rows:
        if r[key_col] == key:
            r[col] = value
            hit += 1
    assert hit == 1, (name, key, hit)
    with open(dst / name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


with tempfile.TemporaryDirectory() as t:
    tmp = Path(t)
    at._build_synthetic(tmp)
    rc, _ = at._quiet(at.analyse, at._ns(tmp, tmp / "out"))
    assert rc == 0, rc
    rc, doc, log = run_indep(tmp, tmp / "out", "ok")
    ok = rc == 0 and doc.get("verdict") == "PASS" and doc.get("contrasts") == 14 and doc.get("cells") == 22 and doc.get("anchors_byte_equal") == 9
    print(("PASS" if ok else "FAIL"), "positive control: agrees with the analysis on the synthetic run", f"(rc {rc}; {log.strip().splitlines()[-1] if log.strip() else ''})")
    if not ok:
        fails.append("positive")
        print(log)
    cases = [("holm_p", "decision_inputs.csv", "criterion_id", "D2:cifar100_full:dann_lrf-naive_branch", "holm_p", "0.5"),
             ("cell_mean", "summary_by_cell.csv", "model", "vann_same", "acc_mean_pp", "99.0"),
             ("anchor", "anchors.csv", "anchor_unit", "cifar_full__stem_mlp__s00", "bytes_equal", "False")]
    for tag, name, key_col, key, col, value in cases:
        dst = tmp / f"out_{tag}"
        if tag == "cell_mean":
            # two cells carry model vann_same (30 and 100 epochs): tamper only the 30-epoch one
            shutil.copytree(tmp / "out", dst)
            with open(dst / name, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                if r["model"] == "vann_same" and r["condition"] == "cifar100_full":
                    r[col] = value
            with open(dst / name, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        else:
            tamper(tmp / "out", dst, name, key_col, key, col, value)
        rc, doc, _ = run_indep(tmp, dst, tag)
        ok = rc == 2 and doc.get("verdict") == "DISAGREE" and len(doc.get("disagreements", [])) == 1
        print(("PASS" if ok else "FAIL"), f"negative control ({tag}): one disagreement, exit 2", f"(rc {rc}; {doc.get('disagreements')})")
        if not ok:
            fails.append(tag)
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
