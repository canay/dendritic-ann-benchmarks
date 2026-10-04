"""Byte-equivalence check: r0 entry point (benchmark.py) vs the R1 runner path.

Runs ``benchmark.py`` itself as a subprocess on CPU for a small configuration,
then runs the same units through ``r1_runner.execute_unit`` (cached-tensor data,
same r0 model/training code) and compares every epoch-history CSV byte for byte.
Exit 0 only when all files are identical. CPU is used because CUDA kernels are
not bitwise reproducible across code paths by construction.

Usage (from the dann_benchmark folder):
    python r1/golden_check.py --data-root <data> --work <scratch dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--dataset", default="fashionmnist")
    ap.add_argument("--subset", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--models", nargs="+", default=["dann_lrf", "naive_branch", "mlp_param", "dann_random"])
    args = ap.parse_args()

    work = Path(args.work)
    r0_out = work / "r0_path"
    r1_out = work / "r1_path"
    work.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(ROOT / "benchmark.py"), "--dataset", args.dataset, "--data-root", args.data_root,
           "--output-dir", str(r0_out), "--models", *args.models, "--seeds", *[str(s) for s in args.seeds],
           "--epochs", str(args.epochs), "--batch-size", "256", "--subset-fraction", str(args.subset),
           "--device", "cpu"]
    print("r0 command:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    if rc != 0:
        print(f"TOOL FAULT: benchmark.py exited {rc}")
        return 2

    import torch

    from r1.r1_runner import execute_unit

    class A:  # minimal args object for execute_unit
        device = "cpu"
        data_root = args.data_root
        run_id = "golden_check"
        worker_index = 0

    state = {"progress": None}
    results = []
    for seed in args.seeds:
        for model in args.models:
            unit = {
                "unit_id": f"golden__{model}__s{seed:02d}", "family": "golden", "condition": "golden",
                "dataset": args.dataset, "subset_fraction": args.subset, "model": model, "seed": seed,
                "epochs": args.epochs, "batch_size": 256, "lr": 1e-3, "val_fraction": 0.1,
                "soma_units": 128, "branches_per_soma": 4, "sample_size": 16, "patch_h": 4, "patch_w": 4,
                "extra": {},
            }
            unit_dir = r1_out / unit["unit_id"]
            if unit_dir.exists():
                print(f"TOOL FAULT: {unit_dir} exists; use a fresh --work folder")
                return 2
            execute_unit(unit, A, state, unit_dir, attempt=1)
            a = r0_out / args.dataset / "histories" / f"{model}_seed{seed}.csv"
            b = unit_dir / "attempt_01" / "history.csv"
            same = a.read_bytes() == b.read_bytes()
            results.append({"model": model, "seed": seed, "r0_sha256": sha(a), "r1_sha256": sha(b), "identical": same})
            print(f"{model:14s} seed {seed}: {'IDENTICAL' if same else 'DIFFERENT'}  {sha(a)[:16]} / {sha(b)[:16]}", flush=True)
    ok = all(r["identical"] for r in results) and len(results) == len(args.seeds) * len(args.models)
    report = {"torch": torch.__version__, "dataset": args.dataset, "subset": args.subset, "epochs": args.epochs,
              "results": results, "verdict": "PASS" if ok else "FAIL"}
    (work / "golden_check_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("verdict:", report["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
