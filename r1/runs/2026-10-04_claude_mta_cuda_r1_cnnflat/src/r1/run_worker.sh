#!/usr/bin/env bash
# One R1 worker under a bounded restart loop.
# Restart ONLY after a per-unit timeout (124); every other non-zero exit stops
# fail-closed. Usage:
#   run_worker.sh <code_dir> <plan> <out_dir> <run_id> <data_root> <worker_index> <num_workers> [max_restarts]
set -u
CODE=$1; PLAN=$2; OUT=$3; RUN_ID=$4; DATA=$5; K=$6; N=$7; MAX=${8:-3}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
mkdir -p "$OUT/logs"
LOG="$OUT/logs/worker_w${K}.log"
cd "$CODE" || exit 2
restarts=0
while true; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start worker=$K/$N restarts=$restarts pid=$$" >> "$LOG"
  python3 r1/r1_runner.py run --plan "$PLAN" --out-dir "$OUT" --run-id "$RUN_ID" --data-root "$DATA" \
    --device cuda --deterministic --threads 2 --worker-index "$K" --num-workers "$N" \
    --heartbeat-seconds 60 --unit-timeout-seconds 1800 --max-attempts 2 \
    ${R1_IDENTITY_FILE:+--identity-file "$R1_IDENTITY_FILE"} >> "$LOG" 2>&1
  RC=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exit rc=$RC" >> "$LOG"
  if [ "$RC" -eq 124 ] && [ "$restarts" -lt "$MAX" ]; then
    restarts=$((restarts + 1))
    continue
  fi
  exit "$RC"
done
