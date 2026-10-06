"""Structural check of the F1 convergence-arm plan (MC-NEURO-R1-004, A18) against the frozen plan; read-only, no training.

    python -B check_plan_f1conv.py

1. configs/plan_f1conv_run.json has 240 units and passes r1/aggregate_f1conv.plan_problems against the frozen plan (every unit
   equals its frozen 30-epoch main-grid counterpart except the unit id, the condition, the family and the number of epochs).
2. The stored tier D plan still equals the tierd_run builder (the builder file was edited to add f1conv_run).
3. Prints the plan sha256 for the identity record. Operation neucom-r1-f1conv-protocol-launch-20261005.
"""
import hashlib
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
sys.path.insert(0, str(PROJECT / "dann_benchmark"))
from r1 import aggregate_f1conv as af  # noqa: E402
from r1 import r1_plan  # noqa: E402
from r1.r1_runner import load_plan  # noqa: E402

plan_p = RUN / "configs" / "plan_f1conv_run.json"
frozen_p = PROJECT / "experiments" / "2026-10-03_claude_mta_cuda_r1_frozen" / "configs" / "plan_frozen_all.json"
tierd_p = PROJECT / "experiments" / "2026-10-04_claude_mta_cuda_r1_tierd" / "configs" / "plan_tierd_run.json"
plan, frozen, tierd = load_plan(plan_p), load_plan(frozen_p), load_plan(tierd_p)
problems = af.plan_problems(plan["units"], {u["unit_id"]: u for u in frozen["units"]})
fams = {u["family"] for u in plan["units"]}
print("units", len(plan["units"]), "families", sorted(fams), "epochs", sorted({u["epochs"] for u in plan["units"]}),
      "run_id", plan["run_id"], "family", plan["family"])
print("plan problems:", len(problems))
for p in problems[:10]:
    print("  ", p)
tierd_equal = tierd["units"] == r1_plan.tierd_run()
print("stored tier D plan equals the tierd_run builder:", tierd_equal)
sha = hashlib.sha256(plan_p.read_bytes()).hexdigest().upper()
print("plan sha256", sha)
ok = len(plan["units"]) == 240 and not problems and tierd_equal and plan["run_id"] == "2026-10-05_claude_mta_cuda_r1_f1conv"
print("CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
