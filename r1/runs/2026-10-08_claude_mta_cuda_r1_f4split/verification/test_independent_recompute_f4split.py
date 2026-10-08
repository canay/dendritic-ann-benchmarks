"""Test of independent_recompute_f4split.py before any unit runs (synthetic outputs; never evidence).

    python -B test_independent_recompute_f4split.py

Builds a synthetic run folder for the real f4split plan (its own writer; valid for the runner's validation), archives it,
runs the real analysis (dann_benchmark/r1/aggregate_f4split.py analyse) and then the recomputation:
  positive control: exit 0 PASS;
  three planted analysis errors (one unit's correct count, the split paired mean, one gate flag): exit 1 DISAGREE each;
  tool fault: an archive missing one planned unit: exit 3.
Exit 0 only if all five behave as expected. Operation neucom-r1-round-e-20261007 (Cowork-Claude, claude-opus-5-5, xhigh).
"""
import csv
import hashlib
import io
import json
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PROJECT = RUN.parents[1]
PLAN = RUN / "configs" / "plan_f4split_run.json"
AGG = PROJECT / "dann_benchmark" / "r1" / "aggregate_f4split.py"
REC = HERE / "independent_recompute_f4split.py"
SPEC_KEYS = ("unit_id", "family", "condition", "dataset", "subset_fraction", "model", "seed", "epochs", "batch_size", "lr",
             "val_fraction", "soma_units", "branches_per_soma", "sample_size", "patch_h", "patch_w", "extra")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()


def write_unit(root: Path, u: dict, rows: list, identity: dict) -> str:
    udir = root / "units" / u["unit_id"] / "attempt_01"
    udir.mkdir(parents=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["dataset", "model_name", "seed", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"])
    for r in rows:
        w.writerow(r)
    data = buf.getvalue().encode("utf-8")
    (udir / "history.csv").write_bytes(data)
    best = min(rows, key=lambda r: float(r[6]))
    spec = {k: u.get(k) for k in SPEC_KEYS}
    res = {"schema_version": 1, "status": "completed", "unit_id": u["unit_id"], "unit_spec": spec,
           "unit_spec_sha256": sha(json.dumps(spec, sort_keys=True, ensure_ascii=True).encode("ascii")), "attempt": 1,
           "history_file": "attempt_01/history.csv", "history_sha256": sha(data), "identity": identity,
           "summary": {"best_val_epoch": int(best[3]), "best_val_loss": float(best[6]), "test_acc_at_best_val": float(best[9]),
                       "test_loss_at_best_val": float(best[8]), "best_test_acc": max(float(r[9]) for r in rows),
                       "final_test_acc": float(rows[-1][9])}}
    (root / "units" / u["unit_id"] / "result.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n",
                                                                encoding="utf-8", newline="\n")
    return sha(data)


def run(cmd: list) -> int:
    return subprocess.run([sys.executable, "-B"] + cmd, capture_output=True).returncode


def main() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    results = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        identity = {"run_id": plan["run_id"], "plan_sha256": sha(PLAN.read_bytes()), "code_fingerprint": "SYNTH",
                    "environment": {}}
        ident_p = td / "IDENTITY.json"
        ident_p.write_text(json.dumps(identity) + "\n", encoding="utf-8")
        root = td / plan["run_id"]
        rng = random.Random(8102026)
        gate_rows, frozen = {}, {}
        for u in plan["units"]:
            ex = u["extra"]
            key = (u["model"], u["seed"]) if u["family"] == "MC005_anchor" else (
                (u["model"], 0) if u["family"] == "MC005_F4_split" and ex["split_seed"] == 0 else None)
            if key is not None and key in gate_rows:
                rows = gate_rows[key]
            else:
                rows = [[u["dataset"], u["model"], u["seed"], e, 2.0 / e, 0.6, 1.5 / e + rng.random() / 5, 0.6, 1.0 / e,
                         (8200 + rng.randrange(0, 1200)) / 10000] for e in range(1, 31)]
                if key is not None:
                    gate_rows[key] = rows
            h = write_unit(root, u, rows, identity)
            if key is not None:
                frozen[f"fashion_full_rand_data__{key[0]}__s{key[1]:02d}"] = h
        rec_p = td / "receipt.json"
        rec_p.write_text(json.dumps({"members": {f"frozen/units/{n}/attempt_01/history.csv": {"sha256": h}
                                                 for n, h in frozen.items()}}), encoding="utf-8")
        arc = td / "outputs.tar.gz"
        with tarfile.open(arc, "w:gz") as tar:
            tar.add(root, arcname=plan["run_id"])
        proc = td / "processed"
        rc = run([str(AGG), "analyse", "--plan", str(PLAN), "--root", str(root), "--identity", str(ident_p),
                  "--frozen-receipt", str(rec_p), "--out", str(proc)])
        assert rc == 0, f"analysis rc {rc}"
        results["positive_control"] = run([str(REC), str(arc), str(PLAN), str(proc), str(rec_p), str(td / "v0.json")])

        def planted(name, fname, mutate):
            bad = td / f"proc_{name}"
            shutil.copytree(proc, bad)
            p = bad / fname
            rows = list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))))
            mutate(rows)
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            results[name] = run([str(REC), str(arc), str(PLAN), str(bad), str(rec_p), str(td / f"v_{name}.json")])

        planted("planted_unit_count", "unit_table.csv", lambda rows: rows[5].update(correct=str(int(rows[5]["correct"]) + 1)))
        planted("planted_paired_mean", "source_paired_summary.csv",
                lambda rows: rows[0].update(diff_mean=repr(float(rows[0]["diff_mean"]) + 0.01)))
        planted("planted_gate_flag", "reproduction_gate.csv", lambda rows: rows[0].update(equal="False"))
        arc2 = td / "missing.tar.gz"
        victim = plan["units"][12]["unit_id"]
        with tarfile.open(arc2, "w:gz") as tar:
            tar.add(root, arcname=plan["run_id"], filter=lambda ti: None if f"/units/{victim}" in ti.name else ti)
        results["tool_fault_missing_unit"] = run([str(REC), str(arc2), str(PLAN), str(proc), str(rec_p), str(td / "v_tf.json")])
    want = {"positive_control": 0, "planted_unit_count": 1, "planted_paired_mean": 1, "planted_gate_flag": 1,
            "tool_fault_missing_unit": 3}
    ok = results == want
    print(json.dumps({"results": results, "expected": want, "pass": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
