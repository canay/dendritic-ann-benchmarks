#!/usr/bin/env bash
# Durability smoke (EXPERIMENT_DURABILITY_AND_RECOVERY.md section 3 and 8.1), NOT manuscript evidence.
#   durability_smoke.sh <code_dir> <data_root> <smoke_dir>
set -u
CODE=$1; DATA=$2; S=$3
mkdir -p "$S"
cd "$CODE" || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
python3 r1/r1_plan.py smoke --run-id dur_smoke --out "$S/plan.json" >/dev/null
RUN="python3 r1/r1_runner.py run --plan $S/plan.json --run-id dur_smoke --data-root $DATA --device cuda --deterministic --threads 2"
echo "== reference (uninterrupted)"
$RUN --out-dir "$S/ref" > "$S/ref.log" 2>&1; echo "ref rc=$?"
echo "== interrupted: stop after 2 units"
$RUN --out-dir "$S/int" --stop-after-units 2 > "$S/int1.log" 2>&1; echo "first rc=$? (expect 75)"
python3 -c "import json;d=json.load(open('$S/int/workers/terminal_status_w0.json'));print('terminal',d['status'],d['exit_code'],'completed',len(d['completed_unit_ids']))"
echo "== resume"
$RUN --out-dir "$S/int" > "$S/int2.log" 2>&1; echo "resume rc=$? (expect 0)"
python3 -c "import json;d=json.load(open('$S/int/workers/terminal_status_w0.json'));print('terminal',d['status'],d['exit_code'],'skipped_validated',len(d['skipped_validated_unit_ids']),'completed',len(d['completed_unit_ids']))"
python3 - "$S" <<'PY'
import sys
from pathlib import Path
S = Path(sys.argv[1])
same = diff = 0
for p in sorted((S / "ref" / "units").glob("*/result.json")):
    uid = p.parent.name
    a = next((S / "ref" / "units" / uid).glob("attempt_*/history.csv"))
    b = next((S / "int" / "units" / uid).glob("attempt_*/history.csv"))
    if a.read_bytes() == b.read_bytes():
        same += 1
    else:
        diff += 1
        print("DIFFERENT", uid)
print(f"interrupted+resumed vs uninterrupted: identical={same} different={diff}")
PY
echo "== heartbeat advancement (1 s cadence, one full-size unit)"
python3 - "$S" <<'PY'
import json, sys
sys.path.insert(0, ".")
from r1.r1_plan import unit
S = sys.argv[1]
plan = {"schema_version": 1, "run_id": "hb_smoke", "family": "smoke", "protocol": "smoke", "base": {},
        "units": [unit("smoke", "hb_fashion_full", "fashionmnist", 1.0, "dann_lrf", 0, epochs=15)]}
open(f"{S}/plan_hb.json", "w").write(json.dumps(plan))
PY
python3 r1/r1_runner.py run --plan "$S/plan_hb.json" --out-dir "$S/hb" --run-id hb_smoke --data-root "$DATA" \
  --device cuda --deterministic --threads 2 --heartbeat-seconds 1 > "$S/hb.log" 2>&1; echo "hb rc=$?"
python3 - "$S" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in (Path(sys.argv[1]) / "hb" / "workers" / "heartbeat_w0.jsonl").read_text().splitlines()]
inunit = [r for r in rows if r.get("unit_id") == "hb_fashion_full__dann_lrf__s00" and r.get("phase") == "train_eval"]
adv = sum(1 for a, b in zip(inunit, inunit[1:]) if b["timestamp"] >= a["timestamp"] and b["process_cpu_seconds"] > a["process_cpu_seconds"] and b["inner_completed"] > a["inner_completed"])
print(f"heartbeats in train_eval: {len(inunit)}; advancing consecutive pairs: {adv}")
print("verdict:", "PASS" if adv >= 2 else "FAIL")
PY
du -sh "$S"
