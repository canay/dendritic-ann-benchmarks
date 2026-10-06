#!/usr/bin/env bash
# Launch N detached, low-priority R1 workers for one plan.
#   launch_batch.sh <code_dir> <plan> <out_dir> <run_id> <data_root> <num_workers> [identity_file]
# With an identity file every worker checks freeze/plan/code/environment identity before any unit runs.
set -u
CODE=$1; PLAN=$2; OUT=$3; RUN_ID=$4; DATA=$5; N=$6
if [ -n "${7:-}" ]; then
  test -f "$7" || { echo "refusing: identity file $7 missing"; exit 2; }
  export R1_IDENTITY_FILE=$7
fi
mkdir -p "$OUT/logs" "$OUT/workers"
if [ -e "$OUT/LAUNCHED" ]; then
  echo "refusing: $OUT/LAUNCHED exists (already launched)"; exit 2
fi
{
  echo "run_id=$RUN_ID"
  echo "launched_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "plan_sha256=$(sha256sum "$PLAN" | cut -d' ' -f1)"
  echo "num_workers=$N"
  echo "identity_file=${R1_IDENTITY_FILE:-none}"
  if [ -n "${R1_IDENTITY_FILE:-}" ]; then echo "identity_sha256=$(sha256sum "$R1_IDENTITY_FILE" | cut -d' ' -f1)"; fi
} > "$OUT/LAUNCHED"
for K in $(seq 0 $((N - 1))); do
  setsid nohup nice -n 10 ionice -c2 -n7 bash "$CODE/r1/run_worker.sh" "$CODE" "$PLAN" "$OUT" "$RUN_ID" "$DATA" "$K" "$N" \
    </dev/null >/dev/null 2>&1 &
  echo "worker_${K}_supervisor_pid=$!" >> "$OUT/LAUNCHED"
done
cat "$OUT/LAUNCHED"
