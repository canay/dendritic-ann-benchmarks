"""Durable F10 timing runner (protocol amendment A11, review PCR-012).

    python r1/f10_timing.py run --plan P --out-dir O --run-id R --data-root D --device cuda|cpu [--deterministic]
        [--threads 2] [--identity-file F] [--heartbeat-seconds 60] [--unit-timeout-seconds 1800] [--max-attempts 2]
        [--stop-after-units N]
    python r1/f10_timing.py verify|status --plan P --out-dir O [--identity-file F]

One timing unit = one (device, dataset, model, seed), measured in a fresh child process so that the host peak RSS
belongs to that unit alone. The child builds the data and the model exactly like the accuracy runner (cached-tensor
loaders; set_seed before the data, the model and the optimiser), then times the training epochs of the unit's schedule
over the fit split (the per-batch operations of ``src.train_eval.train_one_run``: host-to-device copy, zero_grad,
forward, loss, backward, Adam step, loss and accuracy bookkeeping; its validation and test evaluations are NOT inside
the timer) and the inference passes over the 10,000 test images (host-to-device copy, forward, argmax; eval mode,
no_grad). The first epoch and the first pass are untimed warm-ups. On CUDA the device is synchronised before every
timer starts and stops, and the peak-memory counters are reset immediately before the first timed epoch and before
the first timed pass. Every measurement also stores the busy and steal time of the process's CPU set and of the whole
host (/proc/stat) and the process CPU time, so foreign load is visible; no measurement is dropped or repeated.

Durability (EXPERIMENT_DURABILITY_AND_RECOVERY.md): per-unit folder, atomic writes, resume skips units whose
result.json validates, new attempt ids for incomplete attempts, one heartbeat file rewritten every
``--heartbeat-seconds`` with the CPU time of the whole process tree (runner + child), per-unit timeout, terminal status
COMPLETED/0, FAILED/3, CONTROLLED_STOP/75 (``--stop-after-units``, interruption smoke only), CANCELLED/143 or 130.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
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

from r1.r1_runner import (EXIT_CONTROLLED_STOP, EXIT_FAILED, EXIT_OK, EXIT_SIGINT, EXIT_SIGTERM, SCHEMA_VERSION,  # noqa: E402
                          atomic_write_json, bind_output_root, check_identity, environment_record, load_plan,
                          peak_rss_bytes, rss_bytes, unit_spec, unit_spec_sha256, utc_now)

KIND = "f10_timing"


# ---------------------------------------------------------------- host counters
def cpu_times() -> Optional[Dict[int, List[int]]]:
    """Per-CPU jiffies from /proc/stat: [user, nice, system, idle, iowait, irq, softirq, steal]."""
    try:
        with open("/proc/stat", "r", encoding="ascii") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    out: Dict[int, List[int]] = {}
    for line in lines:
        if line.startswith("cpu") and line[3:4].isdigit():
            parts = line.split()
            vals = [int(v) for v in parts[1:9]]
            out[int(parts[0][3:])] = vals + [0] * (8 - len(vals))
    return out


def own_cpu_seconds() -> float:
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF)
    return float(ru.ru_utime + ru.ru_stime)


def contention(before, after, own: float, wall: float, cpus: List[int]) -> Dict[str, Any]:
    if before is None or after is None or wall <= 0:
        return {"available": False}
    hz = float(os.sysconf("SC_CLK_TCK"))

    def busy(v: List[int]) -> int:
        return v[0] + v[1] + v[2] + v[5] + v[6]

    def total(ids) -> tuple:
        ids = [i for i in ids if i in before and i in after]
        return (sum(busy(after[i]) - busy(before[i]) for i in ids) / hz,
                sum(after[i][7] - before[i][7] for i in ids) / hz)

    pin_busy, pin_steal = total(cpus)
    host_busy, host_steal = total(sorted(after))
    return {
        "available": True,
        "pinned_busy_s": round(pin_busy, 3), "pinned_steal_s": round(pin_steal, 3),
        "host_busy_s": round(host_busy, 3), "host_steal_s": round(host_steal, 3), "own_cpu_s": round(own, 3),
        "foreign_load_pinned_cores": round(max(0.0, pin_busy - own) / wall, 4),
        "foreign_load_host_cores": round(max(0.0, host_busy - own) / wall, 4),
        "steal_pinned_cores": round(pin_steal / wall, 4),
    }


def proc_cpu_seconds(pid: Optional[int]) -> Optional[float]:
    if not pid:
        return None
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="ascii", errors="replace") as handle:
            data = handle.read()
    except OSError:
        return None
    fields = data[data.rindex(")") + 2:].split()  # fields[0] is field 3 (state); utime = field 14, stime = field 15
    return (int(fields[11]) + int(fields[12])) / float(os.sysconf("SC_CLK_TCK"))


def proc_rss_bytes(pid: Optional[int]) -> Optional[int]:
    if not pid:
        return None
    try:
        with open(f"/proc/{pid}/status", "r", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


# ---------------------------------------------------------------- measurement (child process)
def train_epoch(model, loader, device, criterion, optimizer, cpus: List[int]) -> Dict[str, Any]:
    import torch

    cuda = device.type == "cuda"
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    if cuda:
        torch.cuda.synchronize()
    s0, c0 = cpu_times(), own_cpu_seconds()
    t0 = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += x.size(0)
    if cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    c1, s1 = own_cpu_seconds(), cpu_times()
    wall = t1 - t0
    return {"seconds": wall, "images": int(total_examples), "train_loss": total_loss / total_examples,
            "train_acc": total_correct / total_examples, "contention": contention(s0, s1, c1 - c0, wall, cpus)}


def inference_pass(model, loader, device, cpus: List[int]) -> Dict[str, Any]:
    import torch

    cuda = device.type == "cuda"
    model.eval()
    n = 0
    predicted = 0
    with torch.no_grad():
        if cuda:
            torch.cuda.synchronize()
        s0, c0 = cpu_times(), own_cpu_seconds()
        t0 = time.perf_counter()
        for x, _y in loader:
            x = x.to(device, non_blocking=True).float()
            pred = model(x).argmax(dim=1)
            n += x.size(0)
            predicted += int(pred.numel())
        if cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        c1, s1 = own_cpu_seconds(), cpu_times()
    if predicted != n:
        raise RuntimeError("prediction count differs from the number of test images")
    wall = t1 - t0
    return {"seconds": wall, "images": int(n), "contention": contention(s0, s1, c1 - c0, wall, cpus)}


def cmd_measure(args) -> int:
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    from r1.f10_macs import closed_form, counted
    from r1.r1_data import load_cached, make_cached_dataloaders
    from r1.r1_runner import build_unit_model
    from src.train_eval import set_seed

    plan = load_plan(Path(args.plan))
    unit = next(u for u in plan["units"] if u["unit_id"] == args.unit_id)
    extra = unit.get("extra") or {}
    sched = extra["timing"]
    unit_dir = Path(args.out_dir) / "units" / unit["unit_id"]
    attempt_dir = Path(args.attempt_dir)
    total_steps = int(sched["warmup_epochs"]) + int(sched["timed_epochs"]) + int(sched["warmup_passes"]) + int(sched["timed_passes"])

    def phase(name: str, done: int) -> None:
        atomic_write_json(attempt_dir / "progress.json", {"phase": name, "phase_started_at": utc_now(), "inner_completed": done,
                                                          "inner_total": total_steps, "pid": os.getpid()})

    try:
        if extra.get("device") != args.device:
            raise RuntimeError(f"unit device {extra.get('device')!r} differs from --device {args.device!r}")
        if args.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
        torch.set_num_threads(int(args.threads))
        device = torch.device(args.device)
        cuda = device.type == "cuda"
        cpus = sorted(os.sched_getaffinity(0))
        seed = int(unit["seed"])
        identity = json.loads(Path(args.identity_file).read_text(encoding="utf-8")) if args.identity_file else None
        t_unit = time.time()

        phase("load_data", 0)
        rss_start = rss_bytes()
        cached = load_cached(unit["dataset"], args.data_root)
        set_seed(seed)
        loaders, info = make_cached_dataloaders(cached, unit["dataset"], unit["batch_size"], unit["val_fraction"],
                                                unit["subset_fraction"], data_seed=seed, num_workers=0)
        rss_data = rss_bytes()

        phase("build_model", 0)
        set_seed(seed)
        model = build_unit_model(unit, info, seed)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_info = None
        if extra.get("r1_model"):
            from r1.r1_models import describe_r1_model

            model_info = describe_r1_model(model)
        macs = {"closed_form": closed_form(unit, info.input_dim, info.num_classes), "counted": counted(model, info.input_dim)}
        set_seed(seed)  # train_one_run seeds again before it builds the optimiser
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=float(unit["lr"]), betas=(0.9, 0.999))
        model.to(device)
        rss_model = rss_bytes()

        done = 0
        train_records = []
        n_epochs = int(sched["warmup_epochs"]) + int(sched["timed_epochs"])
        for e in range(n_epochs):
            timed = e >= int(sched["warmup_epochs"])
            phase(f"train_epoch_{e + 1}" + ("" if timed else "_warmup"), done)
            if cuda and e == int(sched["warmup_epochs"]):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            rec = train_epoch(model, loaders["train"], device, criterion, optimizer, cpus)
            rec.update({"index": e + 1, "timed": timed})
            train_records.append(rec)
            done += 1
        train_mem = ({"max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                      "max_reserved_bytes": int(torch.cuda.max_memory_reserved())} if cuda else None)

        infer_records = []
        n_passes = int(sched["warmup_passes"]) + int(sched["timed_passes"])
        for p in range(n_passes):
            timed = p >= int(sched["warmup_passes"])
            phase(f"inference_pass_{p + 1}" + ("" if timed else "_warmup"), done)
            if cuda and p == int(sched["warmup_passes"]):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            rec = inference_pass(model, loaders["test"], device, cpus)
            rec.update({"index": p + 1, "timed": timed})
            infer_records.append(rec)
            done += 1
        infer_mem = ({"max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                      "max_reserved_bytes": int(torch.cuda.max_memory_reserved())} if cuda else None)
        phase("write_outputs", done)

        result = {
            "schema_version": SCHEMA_VERSION, "status": "completed", "kind": KIND,
            "run_id": args.run_id, "unit_id": unit["unit_id"], "unit_spec": unit_spec(unit),
            "unit_spec_sha256": unit_spec_sha256(unit), "attempt": int(args.attempt),
            "device": args.device, "deterministic": bool(args.deterministic), "threads": int(args.threads),
            "torch_num_threads": torch.get_num_threads(), "cpu_affinity": cpus, "schedule": sched,
            "batch_size": int(unit["batch_size"]), "fit_images": len(loaders["train"].dataset),
            "test_images": len(loaders["test"].dataset),
            "train_measurements": train_records, "inference_measurements": infer_records,
            "peak_cuda_memory_train": train_mem, "peak_cuda_memory_inference": infer_mem,
            "host_rss_bytes": {"start": rss_start, "after_data": rss_data, "after_model": rss_model, "peak": peak_rss_bytes()},
            "trainable_params": int(params), "model_info": model_info, "macs_per_image_forward": macs,
            "identity": identity, "environment": environment_record(args.device), "data_hashes": cached["hashes"],
            "started_at_utc": datetime.fromtimestamp(t_unit, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ended_at_utc": utc_now(), "unit_seconds": round(time.time() - t_unit, 3), "pid": os.getpid(),
        }
        atomic_write_json(attempt_dir / "attempt.json", result)
        atomic_write_json(unit_dir / "result.json", result)
        return EXIT_OK
    except Exception:  # noqa: BLE001
        atomic_write_json(attempt_dir / "attempt.json", {"status": "failed", "unit_id": args.unit_id, "at": utc_now(),
                                                         "traceback": traceback.format_exc()})
        return EXIT_FAILED


# ---------------------------------------------------------------- validation
def validate_timing(unit: Dict[str, Any], unit_dir: Path, identity: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """None when the timing unit is validly complete; otherwise the reason it is not."""
    path = unit_dir / "result.json"
    if not path.is_file():
        return "no result.json"
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"result.json unreadable: {exc}"
    if r.get("schema_version") != SCHEMA_VERSION or r.get("status") != "completed" or r.get("kind") != KIND:
        return "result.json schema/status/kind invalid"
    if r.get("unit_spec_sha256") != unit_spec_sha256(unit):
        return "unit spec changed since the result was written"
    extra = unit.get("extra") or {}
    sched = extra.get("timing") or {}
    if r.get("device") != extra.get("device"):
        return "device differs from the unit"
    if int(r.get("test_images", 0)) != 10000:
        return "test set is not 10,000 images"
    if int(r.get("fit_images", 0)) <= 0:
        return "empty fit split"
    for key, n_timed, n_warm, img_key in (("train_measurements", "timed_epochs", "warmup_epochs", "fit_images"),
                                         ("inference_measurements", "timed_passes", "warmup_passes", "test_images")):
        recs = r.get(key) or []
        if len(recs) != int(sched[n_timed]) + int(sched[n_warm]):
            return f"{key}: {len(recs)} records"
        if sum(1 for m in recs if m.get("timed")) != int(sched[n_timed]):
            return f"{key}: timed count differs from the schedule"
        for m in recs:
            s = m.get("seconds")
            if not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0:
                return f"{key}: non-positive or non-finite seconds"
            if int(m.get("images", -1)) != int(r[img_key]):
                return f"{key}: image count differs from {img_key}"
    for m in r["train_measurements"]:
        if not math.isfinite(float(m.get("train_loss", float("nan")))):
            return "non-finite training loss"
    macs = r.get("macs_per_image_forward") or {}
    if not macs.get("closed_form") or macs.get("closed_form") != macs.get("counted"):
        return "MAC counts of the two methods disagree"
    if not (r.get("host_rss_bytes") or {}).get("peak"):
        return "host peak RSS missing"
    if r.get("device") == "cuda":
        for key in ("peak_cuda_memory_train", "peak_cuda_memory_inference"):
            if not (r.get(key) or {}).get("max_allocated_bytes"):
                return f"{key} missing"
    if identity is not None and r.get("identity") != identity:
        return "identity mismatch (freeze, plan, code or environment)"
    return None


# ---------------------------------------------------------------- runner (parent process)
class TreeHeartbeat(threading.Thread):
    """Heartbeat of the runner AND its measuring child (8.1: the process tree, not the wrapper PID)."""

    def __init__(self, path: Path, state: Dict[str, Any], interval: float) -> None:
        super().__init__(daemon=True)
        self.path, self.state, self.interval = path, state, interval
        self.stop_event = threading.Event()

    def snapshot(self) -> Dict[str, Any]:
        st = dict(self.state)
        child = st.get("child_pid")
        progress = {}
        if st.get("attempt_dir"):
            try:
                progress = json.loads((Path(st["attempt_dir"]) / "progress.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                progress = {}
        wrapper = round(time.process_time(), 2)
        child_cpu = proc_cpu_seconds(child)
        started = st.get("unit_started_epoch")
        return {
            "timestamp": utc_now(), "run_id": st.get("run_id"), "worker": 0, "unit_id": st.get("unit_id"),
            "attempt_id": st.get("attempt_id"), "pid": os.getpid(), "child_pid": child,
            "sampled_pids": [os.getpid()] + ([child] if child else []),
            "phase": progress.get("phase") or st.get("phase"),
            "phase_started_at": progress.get("phase_started_at") or st.get("phase_started_at"),
            "unit_elapsed_seconds": None if started is None else round(time.time() - started, 1),
            "completed_atomic_units": st.get("completed_units"), "planned_atomic_units": st.get("planned_units"),
            "last_durable_checkpoint_at": st.get("last_checkpoint_at"),
            "wrapper_cpu_seconds": wrapper,
            "process_tree_cpu_seconds": round(wrapper + (child_cpu or 0.0), 2),
            "child_cpu_seconds": None if child_cpu is None else round(child_cpu, 2),
            "child_rss_bytes": proc_rss_bytes(child),
            "inner_completed": progress.get("inner_completed"), "inner_total": progress.get("inner_total"),
            "inner_unit": "timing_steps (epochs + inference passes)",
        }

    def write(self) -> None:
        snap = self.snapshot()
        try:
            atomic_write_json(self.path, snap)
            with open(self.path.with_suffix(".jsonl"), "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(snap, sort_keys=True) + "\n")
                handle.flush()
        except OSError:
            pass

    def run(self) -> None:
        self.write()
        while not self.stop_event.wait(self.interval):
            self.write()


def cmd_run(args) -> int:
    plan = load_plan(Path(args.plan))
    out = Path(args.out_dir)
    units_root = out / "units"
    status_root = out / "workers"
    status_root.mkdir(parents=True, exist_ok=True)
    terminal_path = status_root / "terminal_status_w0.json"
    env_record = environment_record(args.device)
    env_record["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    atomic_write_json(status_root / "environment_w0.json", env_record)
    started_at = utc_now()
    identity = None
    problems: List[str] = []
    if any((u.get("extra") or {}).get("device") != args.device for u in plan["units"]):
        problems.append(f"the plan holds units for a device other than {args.device}")
    if args.identity_file and not problems:
        identity = json.loads(Path(args.identity_file).read_text(encoding="utf-8"))
        problems = check_identity(identity, args, env_record)
        if not problems:
            bound = bind_output_root(out, identity)
            if bound:
                problems.append(bound)
    if problems:
        atomic_write_json(terminal_path, {"schema_version": SCHEMA_VERSION, "run_id": args.run_id, "worker": 0,
                                          "status": "FAILED", "exit_code": EXIT_FAILED, "identity_problems": problems,
                                          "ended_at_utc": utc_now(), "pid": os.getpid()})
        print("PRE-RUN CHECK FAILED:", "; ".join(problems), flush=True)
        return EXIT_FAILED

    units = plan["units"]
    state: Dict[str, Any] = {"run_id": args.run_id, "unit_id": None, "attempt_id": None, "phase": "startup",
                             "phase_started_at": utc_now(), "unit_started_epoch": None, "completed_units": 0,
                             "planned_units": len(units), "last_checkpoint_at": None, "child_pid": None, "attempt_dir": None}
    record: Dict[str, List[str]] = {"completed": [], "skipped_validated": [], "failed": [], "timed_out": []}
    current: Dict[str, Any] = {"proc": None}

    def terminal(status: str, code: int) -> None:
        atomic_write_json(terminal_path, {
            "schema_version": SCHEMA_VERSION, "run_id": args.run_id, "worker": 0, "status": status, "exit_code": code,
            "started_at_utc": started_at, "ended_at_utc": utc_now(), "planned_units": len(units),
            "completed_unit_ids": record["completed"], "skipped_validated_unit_ids": record["skipped_validated"],
            "failed_unit_ids": record["failed"], "timed_out_unit_ids": record["timed_out"], "pid": os.getpid()})

    def on_signal(signum, _frame):
        proc = current.get("proc")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        code = EXIT_SIGTERM if signum == signal.SIGTERM else EXIT_SIGINT
        terminal("CANCELLED", code)
        os._exit(code)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    atomic_write_json(terminal_path, {"schema_version": SCHEMA_VERSION, "run_id": args.run_id, "worker": 0,
                                      "status": "RUNNING", "exit_code": None, "started_at_utc": started_at, "pid": os.getpid()})
    heartbeat = TreeHeartbeat(status_root / "heartbeat_w0.json", state, args.heartbeat_seconds)
    heartbeat.start()

    done_this_process = 0
    for unit in units:
        uid = unit["unit_id"]
        unit_dir = units_root / uid
        if validate_timing(unit, unit_dir, identity) is None:
            record["skipped_validated"].append(uid)
            state["completed_units"] += 1
            continue
        attempts = sorted(unit_dir.glob("attempt_*")) if unit_dir.is_dir() else []
        if len(attempts) >= args.max_attempts:
            record["failed"].append(uid)
            continue
        attempt = len(attempts) + 1
        attempt_dir = unit_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        cmd = [sys.executable, str(Path(__file__).resolve()), "measure", "--plan", str(Path(args.plan).resolve()),
               "--unit-id", uid, "--out-dir", str(out.resolve()), "--run-id", args.run_id, "--data-root", args.data_root,
               "--device", args.device, "--threads", str(args.threads), "--attempt", str(attempt),
               "--attempt-dir", str(attempt_dir.resolve())]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.identity_file:
            cmd += ["--identity-file", str(Path(args.identity_file).resolve())]
        state.update(unit_id=uid, attempt_id=attempt, unit_started_epoch=time.time(), phase="measure",
                     phase_started_at=utc_now(), attempt_dir=str(attempt_dir))
        with open(attempt_dir / "child.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
            current["proc"] = proc
            state["child_pid"] = proc.pid
            heartbeat.write()
            try:
                rc = proc.wait(timeout=args.unit_timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                rc = None
                record["timed_out"].append(uid)
                atomic_write_json(attempt_dir / "attempt.json", {"status": "timed_out", "unit_id": uid, "at": utc_now(),
                                                                 "timeout_seconds": args.unit_timeout_seconds})
        current["proc"] = None
        if rc not in (None, 0) and not (attempt_dir / "attempt.json").is_file():
            atomic_write_json(attempt_dir / "attempt.json", {"status": "failed", "unit_id": uid, "at": utc_now(), "child_exit_code": rc})
        if validate_timing(unit, unit_dir, identity) is None:
            record["completed"].append(uid)
            state["completed_units"] += 1
            state["last_checkpoint_at"] = utc_now()
            done_this_process += 1
        else:
            record["failed"].append(uid)
        state.update(unit_id=None, attempt_id=None, unit_started_epoch=None, child_pid=None, attempt_dir=None, phase="idle")
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


def _identity(args) -> Optional[Dict[str, Any]]:
    return json.loads(Path(args.identity_file).read_text(encoding="utf-8")) if getattr(args, "identity_file", None) else None


def cmd_verify(args) -> int:
    plan = load_plan(Path(args.plan))
    out = Path(args.out_dir)
    identity = _identity(args)
    bad = 0
    for unit in plan["units"]:
        reason = validate_timing(unit, out / "units" / unit["unit_id"], identity)
        if reason is not None:
            bad += 1
            print(f"INVALID {unit['unit_id']}: {reason}")
    print(f"verified {len(plan['units']) - bad}/{len(plan['units'])}")
    return 0 if bad == 0 else 2


def cmd_status(args) -> int:
    plan = load_plan(Path(args.plan))
    out = Path(args.out_dir)
    identity = _identity(args)
    valid = sum(1 for u in plan["units"] if validate_timing(u, out / "units" / u["unit_id"], identity) is None)
    report: Dict[str, Any] = {"run_id": plan.get("run_id"), "validated": valid, "planned": len(plan["units"])}
    for name in ("heartbeat_w0.json", "terminal_status_w0.json"):
        try:
            report[name] = json.loads((out / "workers" / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report[name] = None
    if isinstance(report.get("terminal_status_w0.json"), dict):
        ts = report["terminal_status_w0.json"]
        report["terminal_status_w0.json"] = {k: ts.get(k) for k in ("status", "exit_code", "ended_at_utc")}
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--device", required=True, choices=["cuda", "cpu"])
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--deterministic", action="store_true")
    run.add_argument("--heartbeat-seconds", type=float, default=60.0)
    run.add_argument("--unit-timeout-seconds", type=float, default=1800.0)
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--stop-after-units", type=int, default=0)
    run.add_argument("--identity-file", default=None)
    meas = sub.add_parser("measure")
    meas.add_argument("--plan", required=True)
    meas.add_argument("--unit-id", required=True)
    meas.add_argument("--out-dir", required=True)
    meas.add_argument("--run-id", required=True)
    meas.add_argument("--data-root", required=True)
    meas.add_argument("--device", required=True, choices=["cuda", "cpu"])
    meas.add_argument("--threads", type=int, default=2)
    meas.add_argument("--deterministic", action="store_true")
    meas.add_argument("--attempt", type=int, required=True)
    meas.add_argument("--attempt-dir", required=True)
    meas.add_argument("--identity-file", default=None)
    for name in ("verify", "status"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--identity-file", default=None)
    args = parser.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "measure":
        return cmd_measure(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
