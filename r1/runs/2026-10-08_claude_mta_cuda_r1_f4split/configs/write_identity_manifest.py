"""Write configs/IDENTITY.json and RUN_MANIFEST.json for run 2026-10-08_claude_mta_cuda_r1_f4split (MC-NEURO-R1-005).

Every hash is measured here from the files on disk; the environment values are those of the frozen environment of protocol
Section 1 on mta-cuda (the launch preflight compares them with the live host before any unit runs). Refuses to overwrite.
Operation neucom-r1-round-e-20261007 (Cowork-Claude, claude-opus-5-5, xhigh); the manifest carries the clock time of the run.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
RUN_ID = "2026-10-08_claude_mta_cuda_r1_f4split"
OP = "neucom-r1-round-e-20261007"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


plan = RUN / "configs" / "plan_f4split_run.json"
protocol = PROJECT / "MD" / "09_audit_revision" / "R1_EVIDENCE_PROTOCOL.md"
agg = PROJECT / "dann_benchmark" / "r1" / "aggregate_f4split.py"
rec = RUN / "verification" / "independent_recompute_f4split.py"
rec_test = RUN / "verification" / "test_independent_recompute_f4split.py"
check = PROJECT / "MD" / "09_audit_revision" / "R1_F4SPLIT" / "check_f4split.py"
builder = PROJECT / "dann_benchmark" / "r1" / "r1_plan.py"
runner = PROJECT / "dann_benchmark" / "r1" / "r1_runner.py"
receipt = json.loads((RUN / "env" / "transfer_v8" / "TRANSFER_RECEIPT_code.json").read_text(encoding="utf-8"))
tierd_ident = json.loads((PROJECT / "experiments" / "2026-10-04_claude_mta_cuda_r1_tierd" / "configs" / "IDENTITY.json").read_text(encoding="utf-8"))
ledger = (PROJECT / "MD" / "_state" / "METHODOLOGY_CHANGE_LEDGER.md").read_text(encoding="utf-8")
lock_lines = [l for l in ledger.splitlines() if l.startswith("Protocol lock: MC-NEURO-R1-005 ")]
assert len(lock_lines) == 1, lock_lines
lock_sha = lock_lines[0].split("sha256=")[1].split()[0].upper()
assert lock_sha == sha(protocol), "the protocol changed after its lock"
assert receipt["code_fingerprint"] == "7A2716FE18A231F61ACDE0E5208699BA3387615D7D331BC0D09A5CCAF05C287B"
assert receipt["archives"]["code_snapshot.tar.gz"]["sha256"] == "ED355051F86ED25FE366DCDB1F434BB465BB58E953F4A90255472A926C9C061B"
smoke_log = RUN / "env" / "host_checks" / "stage_v8_smoke.txt"

identity = {
    "freeze_id": "MC-NEURO-R1-005",
    "change_id": "MC-NEURO-R1-005",
    "amendment": "A20-A21 (DEC-NEURO-038)",
    "run_id": RUN_ID,
    "plan_sha256": sha(plan),
    "code_fingerprint": receipt["code_fingerprint"],  # code snapshot v8, staged and verified on mta-cuda
    "environment": tierd_ident["environment"],  # the frozen environment of Section 1; re-checked live by the launch preflight
    "protocol_sha256": lock_sha,
}
ident_p = RUN / "configs" / "IDENTITY.json"
man_p = RUN / "RUN_MANIFEST.json"
if ident_p.exists() or man_p.exists():
    sys.exit("refusing: IDENTITY.json or RUN_MANIFEST.json exists")
ident_p.write_text(json.dumps(identity, indent=1) + "\n", encoding="utf-8", newline="\n")
R = "<remote-home>/experiments/SCI-reports_bias_dendritic_dac_package/2026-10-03"
manifest = {
    "run_id": RUN_ID,
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "makale_kisa_adi": "dendritic_ann_branching_controlled",
    "operation_id": OP,
    "methodology_change_id": "MC-NEURO-R1-005",
    "decisions": ["DEC-NEURO-038 (R2.14: split-only and order-only arm on mta-cuda)"],
    "protocol": {"path": "MD/09_audit_revision/R1_EVIDENCE_PROTOCOL.md", "sha256_at_lock": lock_sha.lower(),
                 "lock_line": lock_lines[0], "family": "F4 with the data source split: FashionMNIST full data; DANN-LRF and "
                 "Naive-Branch; split seed 0-9 (order 0) and order seed 1-9 (split 0); 8 reproduction anchors; 30 epochs; A20-A21"},
    "status": "prepared (not launched)",
    "run_kind": "canonical (descriptive sensitivity arm answering R2.14; reported whatever the outcome; rules i-vi of Section 12)",
    "tool": "Cowork-Claude",
    "model_or_session": "claude-opus-5-5 (xhigh)",
    "host_label": "mta-cuda",
    "hostname": "<experiment-host>",
    "local_run_folder": f"experiments/{RUN_ID}/",
    "remote_run_folder": f"{R}/runs/{RUN_ID}/",
    "sync_status": "pending (delivery with a receipt written on the host before transfer; local verification before any removal)",
    "platform_envelope": {"os_family": "Linux", "arch": "x86_64", "python": "3.12.3"},
    "working_directory": f"{R}/code_ed355051f86e",
    "command": ("nice -n 10 ionice -c2 -n7 bash r1/run_worker.sh <code_dir> <plan> <out_dir> " + RUN_ID + f" {R}/data <k> 2 "
                "(k = 0, 1; R1_IDENTITY_FILE set)"),
    "launcher": "env/delivery/mta_launch_f4split.sh (hash, identity, environment, code-fingerprint and resource preflight; then "
                "r1/launch_batch.sh with two workers)",
    "script": "r1/r1_runner.py (v8: optional split_seed and order_seed, defaults equal to the data seed); models and data through "
              "the unchanged r0/v7 code path",
    "config": f"configs/plan_f4split_run.json (sha256 {sha(plan)}, 46 units)",
    "hyperparameters": {"epochs": 30, "batch_size": 256, "lr": 0.001, "val_fraction": 0.1, "soma_units": 128, "branches_per_soma": 4,
                        "sample_size": 16, "patch_h": 4, "patch_w": 4},
    "code_snapshot": {"version": "v8", "code_fingerprint": receipt["code_fingerprint"],
                      "archive": "env/transfer_v8/code_snapshot.tar.gz",
                      "archive_sha256": receipt["archives"]["code_snapshot.tar.gz"]["sha256"],
                      "changed_from_v7": receipt["changed_from_base"], "runner_sha256": sha(runner), "plan_builder_sha256": sha(builder),
                      "host_staging_and_smoke": f"env/host_checks/stage_v8_smoke.txt (sha256 {sha(smoke_log)})"},
    "plan_builder": {"path": "dann_benchmark/r1/r1_plan.py (f4split_run)", "sha256": sha(builder),
                     "structural_check": f"MD/09_audit_revision/R1_F4SPLIT/check_f4split.py (sha256 {sha(check)}): stored frozen, CNN-arm, "
                                          "F10, tier D and F1-convergence plans equal the edited builder; 46 units equal their frozen F4 "
                                          "data-seed units except id, family, condition and seed fields; 6/6 gate histories in the "
                                          "frozen receipt; host build of the plan equal to the local build"},
    "dataset": "FashionMNIST (full data; already staged on the host)",
    "split": "90/10 fit/validation from the training partition by the split seed; full 10,000-image test set",
    "seed": "routing 0, initialization 0; split 0-9 with order 0; order 1-9 with split 0; anchors data seed 3 and 7 (old path) and "
            "split = order = 3 and 7 (new path)",
    "baselines_or_methods": ["dann_lrf", "naive_branch"],
    "resource_limits": {"workers_threads": "2 workers x 2 threads", "priority": "nice -n 10, ionice -c2 -n7",
                        "projected_wall_clock": "about 5 min with two workers (frozen F4 units measured 12.5-12.8 s each, 60 units); "
                                                "protocol projection 15-20 min", "ceiling_wall_clock": "2 h from launch",
                        "unit_timeout_s": 1800, "rss_per_worker_ceiling_bytes": 6442450944, "disk_free_floor_gb": 10},
    "durability_preflight": {
        "atomic_unit": "one (condition, model, value) training run", "planned_unit_count": 46,
        "checkpoint_path_and_schema": "units/<unit_id>/result.json + attempt_NN/history.csv (runner schema_version 1)",
        "atomic_write_strategy": "temporary file, fsync, os.replace (r1_runner.atomic_write_text)",
        "resume_command": "the same launch (r1_runner run skips validated units as skipped_validated)",
        "resume_validation_rule": "r1_runner.validate_result (schema, unit-spec sha256, history sha256, rows, finite values, "
                                  "re-derived summary, identity)",
        "interruption_smoke_evidence": "env/host_checks/stage_v8_smoke.txt: --stop-after-units 2 -> CONTROLLED_STOP 75, resume -> 0 "
                                       "with 2 skipped_validated, verify 4/4 (code v8 on the host)",
        "per_unit_timeout": "1800 s (exit 124, bounded restart in run_worker.sh)", "whole_run_watchdog": "2 h ceiling (manual)",
        "eta_basis_and_margin": "measured frozen F4 unit wall time 12.5-12.8 s x 46 / 2 workers; 2 h ceiling",
        "max_workers_and_thread_limits": "2 workers, OMP/MKL/OPENBLAS/NUMEXPR 2 threads",
        "disk_ram_resource_preflight": "launcher: disk > 10 GB free, available memory > 12 GB, no foreign GPU compute process",
        "progress_heartbeat_path_and_stall_threshold": "workers/heartbeat_w<k>.json(l), 60 s cadence; stall = no new checkpoint for "
                                                       "3 unit times with stale heartbeats",
        "heartbeat_cadence_schema_writer_and_atomicity": "r1_runner.Heartbeat thread, atomic snapshot + flushed jsonl line",
        "heartbeat_advancement_smoke_evidence": "smoke heartbeats advance in timestamp, completed units (0->1->2, 0->3->4), mid-unit "
                                                "batch progress and CPU seconds; the smoke's same-unit pair criterion was mis-specified "
                                                "for 1-2 s smoke units (no two beats fall inside one unit) and is recorded as such",
        "opaque_phase_supervisor_sampling_rule": "not applicable: no opaque phase; batch-level progress inside the unit",
        "opaque_runtime_bound_admission": "mode=measured_short; threshold_seconds=300; measured_seconds=12.8; "
                                          "evidence=experiments/2026-10-03_claude_mta_cuda_r1_frozen raw archive (60 F4 result.json "
                                          "timing_seconds.unit_total); overrun=heartbeat_or_fail_closed",
        "raw_output_contract": "per-unit result.json and history.csv; identity bound per unit",
        "aggregate_script_and_inputs": "dann_benchmark/r1/aggregate_f4split.py analyse (plan, extracted run, identity, frozen receipt)",
        "plot_script_and_inputs": "none (no figure; numbers enter text and Table 3)",
        "decision_artifact_path_and_criterion_schema": "processed_outputs/decision_inputs.json (per-source positive count, mean, SD, "
                                                       "minimum; gate pass)",
        "decision_discriminator_statistics_persisted": "yes (per-source positive count and minimum difference; gate flag)",
        "notification_lifecycle": "no Telegram on this host; durable status, logs and terminal_status files",
        "terminal_status_paths": "workers/terminal_status_w0.json, workers/terminal_status_w1.json",
        "predecessor_terminal_status_poll_rule": "not applicable (no predecessor run)",
        "container_image_layer_budget": "not applicable (no container)",
        "phased_engine_schedule_rule": "not applicable (single phase; disk 21 GB free, run output < 50 MB)",
        "recovery_merge_equivalence_gate": "not applicable (one run, no partial merge)",
        "code_snapshot_path_and_sha256": "env/transfer_v8/code_snapshot.tar.gz ED355051F86ED25FE366DCDB1F434BB465BB58E953F4A90255472A926C9C061B",
        "local_delivery_and_verification_rule": "mta_deliver_prep_v2.sh (terminal COMPLETED/0, identity-checked verify, archive + receipt "
                                                "before transfer), scp, r1/verify_delivery.py with negative controls, second hash tool",
        "partial_result_promotion_policy": "prohibited",
    },
    "freeze_binding": {"freeze_id": "MC-NEURO-R1-005 (A12 precedent: the change id is the freeze id; NEURO-R1-FREEZE-001 final and untouched)",
                       "identity_file": f"experiments/{RUN_ID}/configs/IDENTITY.json", "identity_sha256": sha(ident_p)},
    "analysis_bound_before_results": {
        "aggregate_script": "dann_benchmark/r1/aggregate_f4split.py", "aggregate_script_sha256": sha(agg),
        "aggregate_selftest": "12 checks, failed none (standard library only)",
        "independent_recompute_script": "verification/independent_recompute_f4split.py", "independent_recompute_sha256": sha(rec),
        "independent_recompute_test": f"verification/test_independent_recompute_f4split.py (sha256 {sha(rec_test)}): positive control "
                                      "PASS, three planted errors DISAGREE, a missing unit TOOL FAULT"},
    "manuscript_locations": ["NEURO/manuscript-r1: Section 4.3 randomness sentence, Section 5.2 randomness sentence, Table 3 F4 row "
                             "(after the author approves the wording); response R2.14"],
    "notes": "descriptive sensitivity arm; cannot change any frozen verdict",
    "environment": {"os": "Ubuntu 24.04.4 LTS", "shell": "bash", "runtime": "Python 3.12.3 (/usr/bin/python3, user site <remote-home>/.local)",
                    "virtualenv_or_container": "none; existing user-site CUDA environment",
                    "libraries_or_lockfile": "torch 2.13.0+cu130, torchvision 0.28.0+cu130 (frozen environment of protocol Section 1)",
                    "cpu": "Intel i7-14700 (20C/28T)", "ram": "31 GiB", "gpu": "NVIDIA GeForce RTX 5060 8 GB"},
}
man_p.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print("identity sha256", sha(ident_p))
print("plan sha256", identity["plan_sha256"])
print("protocol sha256", lock_sha)
print("aggregate", sha(agg)[:16], "recompute", sha(rec)[:16], "test", sha(rec_test)[:16])
print("RUN_MANIFEST written", man_p, sha(man_p))
