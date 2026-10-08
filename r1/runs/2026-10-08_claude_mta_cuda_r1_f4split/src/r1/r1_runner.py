"""Durable unit runner for the R1 evidence runs.

One atomic unit = one (condition, model, seed[, variant]) training run, executed
through the r0 code path: ``benchmark.build_model`` and
``src.train_eval.train_one_run`` are called unchanged; only the data are served
from the cached-tensor pipeline in ``r1_data.py`` (byte-equivalence checked by
``golden_check.py``).

Durability contract (EXPERIMENT_DURABILITY_AND_RECOVERY.md):
* every unit writes into its own folder; a unit is complete only when
  ``units/<unit_id>/result.json`` exists, validates (schema, unit-spec hash,
  history SHA-256, row count, finite values) and was written atomically;
* resume skips validated units (``skipped_validated``); an incomplete attempt is
  kept and a new attempt id is opened;
* a heartbeat file per worker is rewritten atomically every ``--heartbeat-seconds``
  (default 60 s) with CPU time, RSS and inner batch progress;
* per-unit timeout -> exit 124 after writing TIMED_OUT; SIGTERM/SIGINT -> exit
  143/130 after writing CANCELLED; ``--stop-after-units`` -> CONTROLLED_STOP/75
  (interruption smoke only); any failed unit -> FAILED/3 at the end.

Subcommands: ``run`` (one worker), ``status`` (read-only progress), ``verify``
(read-only validation of every result).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_FAILED = 3
EXIT_CONTROLLED_STOP = 75
EXIT_TIMEOUT = 124
EXIT_SIGTERM = 143
EXIT_SIGINT = 130
UNIT_SPEC_KEYS = (
    "unit_id", "family", "condition", "dataset", "subset_fraction", "model", "seed",
    "epochs", "batch_size", "lr", "val_fraction", "soma_units", "branches_per_soma",
    "sample_size", "patch_h", "patch_w", "extra",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def unit_spec(unit: Dict[str, Any]) -> Dict[str, Any]:
    return {key: unit.get(key) for key in UNIT_SPEC_KEYS}


def unit_spec_sha256(unit: Dict[str, Any]) -> str:
    blob = json.dumps(unit_spec(unit), sort_keys=True, ensure_ascii=True).encode("ascii")
    return hashlib.sha256(blob).hexdigest().upper()


def load_plan(path: Path) -> Dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("plan schema_version mismatch")
    ids = [u["unit_id"] for u in plan["units"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate unit_id in plan")
    return plan


def rss_bytes() -> Optional[int]:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def peak_rss_bytes() -> Optional[int]:
    try:
        import resource  # Linux only

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- identity (review PCR-009)
CODE_SUFFIXES = (".py", ".sh")
IDENTITY_ENV_KEYS = ("python", "torch", "torch_cuda", "torchvision", "os_family", "arch")


def code_fingerprint_from_pairs(pairs: List[tuple]) -> str:
    """sha256 over sorted 'relpath<TAB>SHA256' lines; the same function serves the runner and the launcher."""
    blob = "".join(f"{rel}\t{digest.upper()}\n" for rel, digest in sorted(pairs)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest().upper()


def code_fingerprint(root: Path) -> str:
    pairs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix in CODE_SUFFIXES or path.name == "requirements.txt":
            pairs.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return code_fingerprint_from_pairs(pairs)


def check_identity(identity: Dict[str, Any], args, environment: Dict[str, Any]) -> List[str]:
    problems = []
    if identity.get("run_id") != args.run_id:
        problems.append("identity run_id differs from --run-id")
    if str(identity.get("plan_sha256", "")).upper() != sha256_file(Path(args.plan)):
        problems.append("plan sha256 differs from the identity record")
    if str(identity.get("code_fingerprint", "")).upper() != code_fingerprint(ROOT):
        problems.append("code fingerprint differs from the identity record")
    for key in IDENTITY_ENV_KEYS:
        if (identity.get("environment") or {}).get(key) != environment.get(key):
            problems.append(f"environment {key}: identity {(identity.get('environment') or {}).get(key)!r} != live {environment.get(key)!r}")
    return problems


def bind_output_root(out: Path, identity: Dict[str, Any]) -> Optional[str]:
    """The output root belongs to exactly one identity; foreign or unbound outputs are never resumed."""
    marker = out / "IDENTITY.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != identity:
            return "output root is bound to a different identity; resume from foreign outputs is forbidden"
        return None
    units = out / "units"
    if units.is_dir() and any(units.iterdir()):
        return "output root already holds units but no identity record; resume from unbound outputs is forbidden"
    atomic_write_json(marker, identity)
    return None


# ---------------------------------------------------------------- validation
def validate_result(unit: Dict[str, Any], unit_dir: Path, identity: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """None when the unit is validly complete; otherwise the reason it is not."""
    result_path = unit_dir / "result.json"
    if not result_path.is_file():
        return "no result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"result.json unreadable: {exc}"
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != "completed":
        return "result.json schema/status invalid"
    if result.get("unit_spec_sha256") != unit_spec_sha256(unit):
        return "unit spec changed since the result was written"
    history = unit_dir / str(result.get("history_file", ""))
    if not history.is_file():
        return "history file missing"
    if sha256_file(history) != result.get("history_sha256"):
        return "history sha256 mismatch"
    with open(history, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(unit["epochs"]):
        return f"history rows {len(rows)} != epochs {unit['epochs']}"
    for row in rows:
        for key in ("train_loss", "val_loss", "test_loss", "test_acc", "val_acc", "train_acc"):
            value = float(row[key])
            if not math.isfinite(value):
                return f"non-finite {key}"
    summary = result.get("summary") or {}
    best = min(rows, key=lambda r: float(r["val_loss"]))  # first minimum in epoch order, as train_one_run selects
    if int(summary.get("best_val_epoch", -1)) != int(best["epoch"]):
        return "summary best_val_epoch disagrees with history"
    # review PCR-011: the primary metric and its companions are re-derived from the history and compared exactly
    # (csv writes repr(float), so the round trip is exact)
    derived = {
        "test_acc_at_best_val": float(best["test_acc"]),
        "best_val_loss": float(best["val_loss"]),
        "test_loss_at_best_val": float(best["test_loss"]),
        "best_test_acc": max(float(r["test_acc"]) for r in rows),
        "final_test_acc": float(rows[-1]["test_acc"]),
    }
    for key, value in derived.items():
        if key not in summary or float(summary[key]) != value:
            return f"summary {key} disagrees with the history"
    if not 0.0 <= derived["test_acc_at_best_val"] <= 1.0:
        return "test_acc_at_best_val outside [0, 1]"
    if identity is not None and result.get("identity") != identity:
        return "identity mismatch (freeze, plan, code or environment)"
    return None


# ---------------------------------------------------------------- execution
class CountingLoader:
    """Transparent wrapper: yields exactly the batches of the wrapped DataLoader."""

    def __init__(self, loader, progress: Dict[str, Any]) -> None:
        self.loader = loader
        self.progress = progress

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        self.progress["epochs_started"] += 1
        for item in self.loader:
            yield item
            self.progress["batches_done"] += 1


class Heartbeat(threading.Thread):
    def __init__(self, path: Path, state: Dict[str, Any], interval: float) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.state = state
        self.interval = interval
        self.stop_event = threading.Event()

    def snapshot(self) -> Dict[str, Any]:
        st = dict(self.state)
        now = time.time()
        started = st.get("unit_started_epoch")
        payload = {
            "timestamp": utc_now(),
            "run_id": st.get("run_id"),
            "worker": st.get("worker"),
            "unit_id": st.get("unit_id"),
            "attempt_id": st.get("attempt_id"),
            "pid": os.getpid(),
            "phase": st.get("phase"),
            "phase_started_at": st.get("phase_started_at"),
            "unit_elapsed_seconds": None if started is None else round(now - started, 1),
            "completed_atomic_units": st.get("completed_units"),
            "planned_atomic_units": st.get("planned_units"),
            "last_durable_checkpoint_at": st.get("last_checkpoint_at"),
            "process_cpu_seconds": round(time.process_time(), 2),
            "rss_bytes": rss_bytes(),
            "inner_completed": st["progress"]["batches_done"] if st.get("progress") else None,
            "inner_total": st.get("inner_total"),
            "inner_unit": "train_batches",
        }
        try:
            import torch

            if torch.cuda.is_available():
                payload["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
        except Exception:  # noqa: BLE001
            pass
        return payload

    def write(self) -> None:
        snap = self.snapshot()
        try:
            atomic_write_json(self.path, snap)
            # append-only liveness log (one flushed line per heartbeat)
            with open(self.path.with_suffix(".jsonl"), "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(snap, sort_keys=True) + "\n")
                handle.flush()
        except OSError:
            pass

    def run(self) -> None:
        self.write()
        while not self.stop_event.wait(self.interval):
            self.write()


def environment_record(device_name: str) -> Dict[str, Any]:
    import torch

    record = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "os_family": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": device_name,
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")},
    }
    try:
        import torchvision

        record["torchvision"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        record["torchvision"] = None
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        record["gpu"] = torch.cuda.get_device_name(0)
    try:
        with open("/proc/cpuinfo", "r", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("model name"):
                    record["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        record["cpu_model"] = platform.processor()
    return record


def build_unit_model(unit: Dict[str, Any], info, model_seed: int):
    """Existing r0 models through benchmark.build_model; R1 additions through r1_models."""
    from benchmark import build_model

    extra = unit.get("extra") or {}
    if extra.get("r1_model"):
        from r1 import r1_models

        return r1_models.build_r1_model(unit, info, model_seed)
    return build_model(
        model_name=unit["model"],
        dataset=unit["dataset"],
        input_dim=info.input_dim,
        num_classes=info.num_classes,
        soma_units=unit["soma_units"],
        branches_per_soma=unit["branches_per_soma"],
        sample_size=unit["sample_size"],
        seed=model_seed,
        patch_h=unit["patch_h"],
        patch_w=unit["patch_w"],
    )


def execute_unit(unit: Dict[str, Any], args, state: Dict[str, Any], unit_dir: Path, attempt: int) -> Dict[str, Any]:
    import torch

    from r1.r1_data import load_cached, make_cached_dataloaders
    from src.train_eval import set_seed, train_one_run, write_epoch_history_csv

    extra = unit.get("extra") or {}
    seed = int(unit["seed"])
    routing_seed = int(extra.get("routing_seed", seed))
    init_seed = int(extra.get("init_seed", seed))
    data_seed = int(extra.get("data_seed", seed))
    # MC-NEURO-R1-005 (protocol Section 12, A20): optional split and order seeds. Both default to the data seed, so every
    # earlier unit runs exactly as before. The split seed drives the training subset and the fit/validation split (explicit
    # generators); the order seed drives the minibatch order (the global RNG that train_one_run re-seeds before epoch 1).
    split_seed = int(extra.get("split_seed", data_seed))
    order_seed = int(extra.get("order_seed", data_seed))
    device = torch.device(args.device)
    attempt_dir = unit_dir / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    state.update(phase="load_data", phase_started_at=utc_now())
    cached = load_cached(unit["dataset"], args.data_root)
    pixel_perm = None
    if extra.get("pixel_permutation_seed") is not None:
        g = torch.Generator().manual_seed(int(extra["pixel_permutation_seed"]))
        pixel_perm = torch.randperm(cached["train_x"].shape[1], generator=g)
    set_seed(split_seed)  # benchmark.py calls set_seed(seed) before make_dataloaders
    loaders, info = make_cached_dataloaders(
        cached, unit["dataset"], unit["batch_size"], unit["val_fraction"], unit["subset_fraction"],
        data_seed=split_seed, num_workers=0, pixel_perm=pixel_perm,
    )
    t_data = time.time()

    state.update(phase="build_model", phase_started_at=utc_now())
    set_seed(init_seed)  # benchmark.py calls set_seed(seed) again before build_model
    model = build_unit_model(unit, info, routing_seed)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = None
    if extra.get("r1_model"):
        from r1.r1_models import describe_r1_model

        model_info = describe_r1_model(model)

    progress = {"epochs_started": 0, "batches_done": 0}
    state["progress"] = progress
    state["inner_total"] = len(loaders["train"]) * int(unit["epochs"])
    loaders["train"] = CountingLoader(loaders["train"], progress)
    state.update(phase="train_eval", phase_started_at=utc_now())
    t_train0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, summary = train_one_run(model, loaders, device, epochs=int(unit["epochs"]), lr=float(unit["lr"]), seed=order_seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train1 = time.time()
    summary.dataset = unit["dataset"]
    summary.model_name = unit["model"]

    state.update(phase="write_outputs", phase_started_at=utc_now())
    hist_tmp = attempt_dir / "history.csv.tmp"
    write_epoch_history_csv(hist_tmp, unit["dataset"], unit["model"], seed, history)
    hist_path = attempt_dir / "history.csv"
    os.replace(hist_tmp, hist_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": args.run_id,
        "unit_id": unit["unit_id"],
        "unit_spec": unit_spec(unit),
        "unit_spec_sha256": unit_spec_sha256(unit),
        "attempt": attempt,
        "history_file": f"attempt_{attempt:02d}/history.csv",
        "history_sha256": sha256_file(hist_path),
        "summary": {
            "trainable_params": int(summary.trainable_params),
            "trainable_params_counted_here": int(trainable),
            "best_val_epoch": int(summary.best_val_epoch),
            "best_val_loss": float(summary.best_val_loss),
            "test_acc_at_best_val": float(summary.test_acc_at_best_val),
            "test_loss_at_best_val": float(summary.test_loss_at_best_val),
            "best_test_acc": float(summary.best_test_acc),
            "min_test_loss": float(summary.min_test_loss),
            "final_test_acc": float(summary.final_test_acc),
            "final_test_loss": float(summary.final_test_loss),
        },
        "model_info": model_info,
        "identity": getattr(args, "identity_record", None),
        "seeds": dict({"seed": seed, "routing_seed": routing_seed, "init_seed": init_seed, "data_seed": data_seed},
                      **({"split_seed": split_seed, "order_seed": order_seed}
                         if ("split_seed" in extra or "order_seed" in extra) else {})),
        "data_hashes": cached["hashes"],
        "timing_seconds": {
            "load_data": round(t_data - t0, 3),
            "train_eval_total": round(t_train1 - t_train0, 3),
            "unit_total": round(time.time() - t0, 3),
        },
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "peak_rss_bytes": peak_rss_bytes(),
        "worker": args.worker_index,
        "pid": os.getpid(),
        "started_at_utc": datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_at_utc": utc_now(),
    }
    atomic_write_json(attempt_dir / "attempt.json", result)
    atomic_write_json(unit_dir / "result.json", result)
    return result


def cmd_run(args) -> int:
    import torch

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if args.threads:
        torch.set_num_threads(int(args.threads))

    plan = load_plan(Path(args.plan))
    out = Path(args.out_dir)
    units_root = out / "units"
    status_root = out / "workers"
    status_root.mkdir(parents=True, exist_ok=True)
    mine = [u for i, u in enumerate(plan["units"]) if i % args.num_workers == args.worker_index]
    terminal_path = status_root / f"terminal_status_w{args.worker_index}.json"
    heartbeat_path = status_root / f"heartbeat_w{args.worker_index}.json"
    env_path = status_root / f"environment_w{args.worker_index}.json"
    env_record = environment_record(args.device)
    atomic_write_json(env_path, env_record)
    identity = None
    if args.identity_file:
        identity = json.loads(Path(args.identity_file).read_text(encoding="utf-8"))
        problems = check_identity(identity, args, env_record)
        if not problems:
            bound = bind_output_root(out, identity)
            if bound:
                problems.append(bound)
        if problems:
            atomic_write_json(terminal_path, {"schema_version": SCHEMA_VERSION, "run_id": args.run_id, "worker": args.worker_index,
                                              "status": "FAILED", "exit_code": EXIT_FAILED, "identity_problems": problems,
                                              "ended_at_utc": utc_now(), "pid": os.getpid()})
            print("IDENTITY CHECK FAILED:", "; ".join(problems), flush=True)
            return EXIT_FAILED
    args.identity_record = identity

    state: Dict[str, Any] = {
        "run_id": args.run_id, "worker": args.worker_index, "unit_id": None, "attempt_id": None,
        "phase": "startup", "phase_started_at": utc_now(), "unit_started_epoch": None,
        "completed_units": 0, "planned_units": len(mine), "last_checkpoint_at": None,
        "progress": None, "inner_total": None,
    }
    record = {"completed": [], "skipped_validated": [], "failed": [], "timed_out": []}
    started_at = utc_now()

    def terminal(status: str, code: int) -> None:
        atomic_write_json(terminal_path, {
            "schema_version": SCHEMA_VERSION, "run_id": args.run_id, "worker": args.worker_index,
            "status": status, "exit_code": code, "started_at_utc": started_at, "ended_at_utc": utc_now(),
            "planned_units": len(mine), "completed_unit_ids": record["completed"],
            "skipped_validated_unit_ids": record["skipped_validated"],
            "failed_unit_ids": record["failed"], "timed_out_unit_ids": record["timed_out"],
            "pid": os.getpid(),
        })

    def on_signal(signum, _frame):
        code = EXIT_SIGTERM if signum == signal.SIGTERM else EXIT_SIGINT
        terminal("CANCELLED", code)
        os._exit(code)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    atomic_write_json(terminal_path, {"schema_version": SCHEMA_VERSION, "run_id": args.run_id,
                                      "worker": args.worker_index, "status": "RUNNING",
                                      "exit_code": None, "started_at_utc": started_at, "pid": os.getpid()})

    heartbeat = Heartbeat(heartbeat_path, state, args.heartbeat_seconds)
    heartbeat.start()

    def watchdog() -> None:
        while True:
            time.sleep(15)
            started = state.get("unit_started_epoch")
            if started is not None and time.time() - started > args.unit_timeout_seconds:
                uid = state.get("unit_id")
                record["timed_out"].append(uid)
                unit_dir = units_root / str(uid)
                atomic_write_json(unit_dir / f"attempt_{int(state.get('attempt_id') or 0):02d}" / "attempt.json",
                                  {"status": "timed_out", "unit_id": uid, "at": utc_now(),
                                   "timeout_seconds": args.unit_timeout_seconds})
                terminal("TIMED_OUT", EXIT_TIMEOUT)
                os._exit(EXIT_TIMEOUT)

    threading.Thread(target=watchdog, daemon=True).start()

    done_this_process = 0
    for unit in mine:
        uid = unit["unit_id"]
        unit_dir = units_root / uid
        reason = validate_result(unit, unit_dir, identity)
        if reason is None:
            record["skipped_validated"].append(uid)
            state["completed_units"] += 1
            continue
        attempts = sorted(unit_dir.glob("attempt_*")) if unit_dir.is_dir() else []
        if len(attempts) >= args.max_attempts:
            record["failed"].append(uid)
            continue
        attempt = len(attempts) + 1
        state.update(unit_id=uid, attempt_id=attempt, unit_started_epoch=time.time(), progress=None, inner_total=None)
        try:
            execute_unit(unit, args, state, unit_dir, attempt)
        except Exception:  # noqa: BLE001
            unit_dir.mkdir(parents=True, exist_ok=True)
            (unit_dir / f"attempt_{attempt:02d}").mkdir(parents=True, exist_ok=True)
            atomic_write_json(unit_dir / f"attempt_{attempt:02d}" / "attempt.json",
                              {"status": "failed", "unit_id": uid, "at": utc_now(),
                               "traceback": traceback.format_exc()})
            record["failed"].append(uid)
            state.update(unit_started_epoch=None, phase="idle")
            continue
        check = validate_result(unit, unit_dir, identity)
        if check is not None:
            record["failed"].append(uid)
        else:
            record["completed"].append(uid)
            state["completed_units"] += 1
            state["last_checkpoint_at"] = utc_now()
            done_this_process += 1
        state.update(unit_started_epoch=None, phase="idle", unit_id=None)
        heartbeat.write()
        if args.stop_after_units and done_this_process >= args.stop_after_units:
            terminal("CONTROLLED_STOP", EXIT_CONTROLLED_STOP)
            heartbeat.stop_event.set()
            return EXIT_CONTROLLED_STOP

    heartbeat.stop_event.set()
    if record["failed"]:
        terminal("FAILED", EXIT_FAILED)
        return EXIT_FAILED
    terminal("COMPLETED", EXIT_OK)
    return EXIT_OK


def _identity_arg(args) -> Optional[Dict[str, Any]]:
    path = getattr(args, "identity_file", None)
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def cmd_status(args) -> int:
    plan = load_plan(Path(args.plan))
    out = Path(args.out_dir)
    identity = _identity_arg(args)
    total = len(plan["units"])
    valid = 0
    invalid: List[str] = []
    for unit in plan["units"]:
        reason = validate_result(unit, out / "units" / unit["unit_id"], identity)
        if reason is None:
            valid += 1
        elif (out / "units" / unit["unit_id"]).is_dir():
            invalid.append(f"{unit['unit_id']}: {reason}")
    report = {"run_id": plan.get("run_id"), "validated": valid, "planned": total,
              "percent": round(100.0 * valid / total, 2) if total else None,
              "started_not_validated": invalid[:20], "workers": {}}
    for hb in sorted((out / "workers").glob("heartbeat_w*.json")):
        try:
            report["workers"][hb.name] = json.loads(hb.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report["workers"][hb.name] = "unreadable"
    for ts in sorted((out / "workers").glob("terminal_status_w*.json")):
        try:
            data = json.loads(ts.read_text(encoding="utf-8"))
            report["workers"][ts.name] = {k: data.get(k) for k in ("status", "exit_code", "ended_at_utc")}
        except (OSError, ValueError):
            report["workers"][ts.name] = "unreadable"
    print(json.dumps(report, indent=2))
    return 0


def cmd_verify(args) -> int:
    plan = load_plan(Path(args.plan))
    out = Path(args.out_dir)
    identity = _identity_arg(args)
    bad = 0
    for unit in plan["units"]:
        reason = validate_result(unit, out / "units" / unit["unit_id"], identity)
        if reason is not None:
            bad += 1
            print(f"INVALID {unit['unit_id']}: {reason}")
    print(f"verified {len(plan['units']) - bad}/{len(plan['units'])}")
    return 0 if bad == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--worker-index", type=int, default=0)
    run.add_argument("--num-workers", type=int, default=1)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--deterministic", action="store_true")
    run.add_argument("--heartbeat-seconds", type=float, default=60.0)
    run.add_argument("--unit-timeout-seconds", type=float, default=1800.0)
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--stop-after-units", type=int, default=0)
    run.add_argument("--identity-file", default=None, help="freeze/plan/code/environment identity; required for frozen runs")
    for name in ("status", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--identity-file", default=None)
    args = parser.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "status":
        return cmd_status(args)
    return cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
