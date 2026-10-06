"""Write configs/IDENTITY.json and RUN_MANIFEST.json for run 2026-10-05_claude_mta_cuda_r1_f1conv (MC-NEURO-R1-004).

Every hash is measured here from the files on disk; the environment values are those of the frozen environment of protocol
Section 1 on mta-cuda (the launch preflight compares them with the live host before any unit runs). Refuses to overwrite.
Operation neucom-r1-f1conv-protocol-launch-20261005 (Cowork-Claude, claude-opus-5-5, max).
"""
import hashlib
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
RUN_ID = "2026-10-05_claude_mta_cuda_r1_f1conv"
OP = "neucom-r1-f1conv-protocol-launch-20261005"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


plan = RUN / "configs" / "plan_f1conv_run.json"
protocol = PROJECT / "MD" / "09_audit_revision" / "R1_EVIDENCE_PROTOCOL.md"
agg = PROJECT / "dann_benchmark" / "r1" / "aggregate_f1conv.py"
stats_src = PROJECT / "dann_benchmark" / "r1" / "aggregate_r1.py"
rec = RUN / "verification" / "independent_recompute_f1conv.py"
rec_test = RUN / "verification" / "test_independent_recompute_f1conv.py"
check = RUN / "configs" / "check_plan_f1conv.py"
builder = PROJECT / "dann_benchmark" / "r1" / "r1_plan.py"
tierd_ident = json.loads((PROJECT / "experiments" / "2026-10-04_claude_mta_cuda_r1_tierd" / "configs" / "IDENTITY.json").read_text(encoding="utf-8"))
ledger = (PROJECT / "MD" / "_state" / "METHODOLOGY_CHANGE_LEDGER.md").read_text(encoding="utf-8")
lock_lines = [l for l in ledger.splitlines() if l.startswith("Protocol lock: MC-NEURO-R1-004 ")]
assert len(lock_lines) == 1, lock_lines
lock_sha = lock_lines[0].split("sha256=")[1].split()[0].upper()
assert lock_sha == sha(protocol), "the protocol changed after its lock"

identity = {
    "freeze_id": "MC-NEURO-R1-004",
    "change_id": "MC-NEURO-R1-004",
    "amendment": "A18-A19 (DEC-NEURO-021)",
    "run_id": RUN_ID,
    "plan_sha256": sha(plan),
    "code_fingerprint": tierd_ident["code_fingerprint"],  # code snapshot v7 on mta-cuda, unchanged (A18 Host and code)
    "environment": tierd_ident["environment"],  # the frozen environment of Section 1; re-checked live by the launch preflight
    "protocol_sha256": lock_sha,
}
assert identity["code_fingerprint"] == "3F58C2F17EB944F7409AD3B7021919CB60996ABC530A77742F3BEC10E2D83BD0"
ident_p = RUN / "configs" / "IDENTITY.json"
man_p = RUN / "RUN_MANIFEST.json"
if ident_p.exists() or man_p.exists():
    sys.exit("refusing: IDENTITY.json or RUN_MANIFEST.json exists")
ident_p.write_text(json.dumps(identity, indent=1) + "\n", encoding="utf-8", newline="\n")

manifest = {
    "run_id": RUN_ID,
    "makale_kisa_adi": "dendritic_ann_branching_controlled",
    "operation_id": OP,
    "methodology_change_id": "MC-NEURO-R1-004",
    "decisions": ["DEC-NEURO-021 (F1 convergence arm at 100 epochs; the frozen 30-epoch labels stay primary)"],
    "protocol": {"path": "MD/09_audit_revision/R1_EVIDENCE_PROTOCOL.md", "sha256_at_lock": lock_sha.lower(),
                 "lock_line": lock_lines[0], "family": "F1 convergence arm: full-data FashionMNIST, KMNIST, CIFAR-10; DANN-LRF, "
                 "Naive-Branch, MLP-Param, DANN-RANDOM; seeds 0-19; 100 epochs; amendments A18-A19"},
    "status": "prepared (not launched)",
    "run_kind": "canonical (budget-sensitivity arm of F1, reported whatever the outcome; interpretation rules i-viii of protocol Section 11)",
    "tool": "Cowork-Claude",
    "model_or_session": "claude-opus-5-5 (max)",
    "host_label": "mta-cuda",
    "hostname": "<experiment-host>",
    "local_run_folder": f"experiments/{RUN_ID}/",
    "remote_run_folder": f"<remote-home>/experiments/SCI-reports_bias_dendritic_dac_package/2026-10-03/runs/{RUN_ID}/",
    "sync_status": "pending (delivery with a receipt written on the host before transfer; local verification before any removal)",
    "platform_envelope": {"os_family": "Linux", "arch": "x86_64", "python": "3.12.3"},
    "working_directory": "<remote-home>/experiments/SCI-reports_bias_dendritic_dac_package/2026-10-03/code_00983a91716f",
    "command": ("nice -n 10 ionice -c2 -n7 bash r1/run_worker.sh <code_dir> <plan> <out_dir> " + RUN_ID +
                " <remote-home>/experiments/SCI-reports_bias_dendritic_dac_package/2026-10-03/data <k> 2 (k = 0, 1; R1_IDENTITY_FILE set)"),
    "launcher": "env/delivery/mta_launch_f1conv.sh (hash, identity, environment, code-fingerprint, KMNIST and resource preflight; "
                "then r1/launch_batch.sh with two workers)",
    "script": "r1/r1_runner.py (unchanged since v4); models and data through the r0 code path (r1_models.py and r1_data.py of v7)",
    "config": f"configs/plan_f1conv_run.json (sha256 {sha(plan)}, 240 units)",
    "code_snapshot": {"version": "v7 (unchanged; already staged)", "code_fingerprint": identity["code_fingerprint"],
                      "archive": "experiments/2026-10-04_claude_mta_cuda_r1_tierd/env/transfer_v7/code_snapshot.tar.gz",
                      "archive_sha256": "00983A91716F7D93955405EF69005779CDBB31B41CCD2E6C5DEB3A3C463588D6"},
    "plan_builder": {"path": "dann_benchmark/r1/r1_plan.py (f1conv_run; local only, not needed on the host)", "sha256": sha(builder),
                     "structural_check": f"configs/check_plan_f1conv.py (sha256 {sha(check)}): 240 units, 0 problems against the frozen plan; "
                                          "stored tier D plan equals its builder"},
    "dataset": "FashionMNIST, KMNIST and CIFAR-10 (full data); KMNIST restaged from dann_benchmark/data/KMNIST (six files, sha256 "
               "checked on the host)",
    "split": "90/10 fit/validation from the training partition by the data seed; full 10,000-image test set",
    "seed": "0-19",
    "hyperparameters": {"epochs": 100, "batch_size": 256, "lr": 0.001, "val_fraction": 0.1, "soma_units": 128,
                        "branches_per_soma": 4, "sample_size": 16, "patch_h": 4, "patch_w": 4},
    "baselines_or_methods": ["dann_lrf", "naive_branch", "mlp_param", "dann_random"],
    "resource_limits": {"workers_threads": "2 workers x 2 threads", "priority": "nice -n 10, ionice -c2 -n7",
                        "projected_wall_clock": "about 1.5-2 h with two workers (projection)", "ceiling_wall_clock": "6 h from launch (hard stop 12 h)",
                        "unit_timeout_s": 1800, "rss_per_worker_ceiling_bytes": 6442450944, "disk_free_floor_gb": 10},
    "freeze_binding": {"freeze_id": "MC-NEURO-R1-004 (A12 precedent: the change id is the freeze id; NEURO-R1-FREEZE-001 final and untouched)",
                       "identity_file": f"experiments/{RUN_ID}/configs/IDENTITY.json", "identity_sha256": sha(ident_p)},
    "analysis_bound_before_results": {
        "aggregate_script": "dann_benchmark/r1/aggregate_f1conv.py", "aggregate_script_sha256": sha(agg),
        "aggregate_selftest": "90 checks, FAILED none (2026-10-05, pinned versions Python 3.12.12, NumPy 2.3.5, SciPy 1.18.0)",
        "statistics_source": "dann_benchmark/r1/aggregate_r1.py", "statistics_source_sha256": sha(stats_src),
        "independent_recompute_script": "verification/independent_recompute_f1conv.py", "independent_recompute_sha256": sha(rec),
        "independent_recompute_test": f"verification/test_independent_recompute_f1conv.py (sha256 {sha(rec_test)}): positive control PASS, "
                                      "three negative controls DISAGREE as planted, tool-fault case exit 3"},
    "manuscript_locations": ["NEURO/manuscript-r1: budget-sensitivity rows or paragraph for RQ1/RQ2 (placement decided after the outcome); "
                             "Discussion, Limitations, future work; response R2-9"],
    "notes": "budget-sensitivity arm of F1; cannot change any NEURO-R1-FREEZE-001 verdict; the 30-epoch labels stay primary",
    "environment": {"os": "Ubuntu 24.04.4 LTS", "shell": "bash", "runtime": "Python 3.12.3 (/usr/bin/python3, user site <remote-home>/.local)",
                    "virtualenv_or_container": "none; existing user-site CUDA environment (TOOL_ENVIRONMENT.md MTA-cuda Environment rule)",
                    "libraries_or_lockfile": "torch 2.13.0+cu130, torchvision 0.28.0+cu130 (frozen environment of protocol Section 1)",
                    "cpu": "Intel i7-14700 (20C/28T)", "ram": "31 GiB", "gpu": "NVIDIA GeForce RTX 5060 8 GB"},
}
man_p.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print("identity sha256", sha(ident_p))
print("plan sha256", identity["plan_sha256"])
print("protocol sha256", lock_sha)
print("aggregate", sha(agg)[:16], "recompute", sha(rec)[:16], "test", sha(rec_test)[:16])
print("RUN_MANIFEST written", man_p)
