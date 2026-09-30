#!/usr/bin/env python3
"""Execute the frozen V12 experiment (CPU only) and audits, never opening fresh data.

Infrastructure failures are retried from scratch within one total budget per run, counted
across runner invocations from the archived attempts: signal-terminated (native) processes,
and CUDA resource-allocation errors identified by fixed log signatures. A failed attempt is
archived and never selected or analyzed. Any other Python error, including gate refusals, is
never retried. A failed primary (C0/C1) run stops the experiment; a failed secondary run is
recorded and excluded by the evaluator. Seeds run sequentially and the variants of one seed
concurrently; every (re)start waits for the frozen free-GPU-memory headroom. The process
must be pinned to the frozen CPU cores (taskset -c 8-23); children inherit the affinity.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.small_training import write_json

_LOCK = threading.Lock()
CUDA_RESOURCE_SIGNATURES = (
    "CUDA out of memory",
    "CUDA error: out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUSOLVER_STATUS_INTERNAL_ERROR",
    "cudaErrorMemoryAllocation",
)


def failure_kind(returncode, log):
    """Classify a failed attempt; only infrastructure failures are retryable."""
    if returncode < 0:
        return "native_crash"
    text = Path(log).read_text(errors="replace")[-20000:]
    if any(signature in text for signature in CUDA_RESOURCE_SIGNATURES):
        return "cuda_resource"
    return None


def archive_attempt(root, label, attempt, returncode, archive, kind):
    """Move a failed infrastructure attempt aside with an incident record."""
    root = Path(root)
    failed = root / "audit" / "failed_attempts" / f"{label}_attempt{attempt}"
    failed.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in (root / "audit" / f"{label}.log", *(Path(p) for p in archive)):
        if path.exists():
            shutil.move(str(path), str(failed / path.name))
            moved.append(str(path))
    with _LOCK:
        record = root / "audit" / "incidents.json"
        incidents = read(record) if record.exists() else []
        incidents.append(
            {
                "label": label,
                "attempt": attempt,
                "kind": kind,
                "returncode": returncode,
                "archived_to": str(failed),
                "moved": moved,
                "unix_time": time.time(),
                "policy": "infrastructure failure: archive, rerun identical command from scratch",
            }
        )
        write_json(record, incidents)


def execute(root, label, args, *, env, times, archive=(), retries=1, before_attempt=None):
    """Run one frozen phase; retry only infrastructure failures, within one total budget."""
    root = Path(root)
    failed = root / "audit" / "failed_attempts"
    prior = len(list(failed.glob(f"{label}_attempt*"))) if failed.exists() else 0
    if prior > retries:
        raise PermissionError(f"{label}: infrastructure retry budget already spent")
    for attempt in range(prior + 1, retries + 2):
        if before_attempt is not None:
            before_attempt(label, attempt)
        print(label, "START attempt", attempt, flush=True)
        tick = time.perf_counter()
        log = root / "audit" / f"{label}.log"
        with log.open("x") as handle:
            process = subprocess.run(
                [sys.executable, *args], stdout=handle, stderr=subprocess.STDOUT, env=env
            )
        if process.returncode == 0:
            times[label] = time.perf_counter() - tick
            print(label, "PASS", times[label], flush=True)
            return
        kind = failure_kind(process.returncode, log)
        if kind is None:
            raise subprocess.CalledProcessError(process.returncode, args)
        archive_attempt(root, label, attempt, process.returncode, archive, kind)
        print(label, kind.upper(), process.returncode, "archived", flush=True)
    raise subprocess.CalledProcessError(process.returncode, args)


def free_gpu_mib():
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return min(int(line) for line in output.split())


def wait_for_gpu(min_free_mib, *, poll_seconds=60, timeout_seconds=6 * 3600, probe=free_gpu_mib):
    """Block until the shared GPU has the frozen free-memory headroom."""
    start = time.time()
    while True:
        free = probe()
        if free >= min_free_mib:
            return {"free_mib": free, "waited_seconds": time.time() - start}
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"GPU free memory stayed below {min_free_mib} MiB")
        print("WAIT_GPU_MEMORY free_mib", free, flush=True)
        time.sleep(poll_seconds)


def train_all(root, config, env, times, gate=None):
    """Every variant and seed concurrently (one core each); only primary failures stop."""
    failures = {}
    pending = [
        (variant, seed)
        for seed in config["seeds"]
        for variant in config["variants"]
        if not (root / "checkpoints" / f"{variant}_{seed}" / "summary.json").exists()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["parallel_workers"]) as pool:
        futures = {
            pool.submit(
                execute,
                root,
                f"train_{variant}_{seed}",
                [
                    "scripts/train_rgbd_width_carrier.py",
                    "--root",
                    str(root),
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--device",
                    "cpu",
                ],
                env=env,
                times=times,
                archive=[root / "checkpoints" / f"{variant}_{seed}"],
                retries=config["infrastructure_retries"],
                before_attempt=gate,
            ): (variant, seed)
            for variant, seed in pending
        }
        concurrent.futures.wait(futures)
    for future, (variant, seed) in futures.items():
        error = future.exception()
        if error is not None:
            failures[f"{variant}_{seed}"] = repr(error)
            print("TRAINING_FAILED", variant, seed, repr(error), flush=True)
    if failures:
        record = root / "audit" / "training_failures.json"
        write_json(record, {**(read(record) if record.exists() else {}), **failures})
    primary = sorted(k for k in failures if k.split("_")[0] in config["primary_pair"])
    if primary:
        raise RuntimeError(f"Primary training failed; experiment stops: {primary}")


def run(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    _, config = verify_lock(root)
    if read(root / "tests.json")["status"] != "PASS":
        raise PermissionError("Full suite must PASS before training")
    if not os.sched_getaffinity(0) <= set(config["cpu_affinity"]):
        raise PermissionError("Run must be pinned to the frozen CPU cores (taskset)")
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "",
        **config["thread_environment"],
    }
    times = {}
    start = time.perf_counter()

    def gate(label, attempt):
        info = {"device": "cpu", "gpu_gate": "SKIPPED_CPU_ONLY_GPU_PAUSED_BY_USER"}
        with _LOCK, (root / "audit" / "gpu_gate.jsonl").open("a") as handle:
            handle.write(json.dumps({"label": label, "attempt": attempt, **info}) + "\n")

    train_all(root, config, env, times, gate)
    if not (root / "audit" / "dev_state_use.json").exists():
        execute(
            root,
            "selected_dev",
            ["scripts/evaluate_rgbd_width_carrier.py", "--root", str(root), "--device", "cpu"],
            env=env,
            times=times,
            archive=[root / "raw" / "selected_dev", root / "raw" / "observability"],
            retries=config["infrastructure_retries"],
            before_attempt=gate,
        )
    execute(
        root,
        "state_correlation",
        ["scripts/diagnose_rgbd_width_states.py", "--root", str(root)],
        env=env,
        times=times,
        retries=config["infrastructure_retries"],
    )
    wall = root / "audit/phase_wall_times.json"
    previous = read(wall) if wall.exists() else {"phases": {}, "invocations": []}
    write_json(
        wall,
        {
            "phases": {**previous["phases"], **times},
            "invocations": [*previous.get("invocations", []), time.perf_counter() - start],
            "parallel_workers": config["parallel_workers"],
            "shared_gpu": True,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    )
    for label, script in (
        ("geometry_free_reference", "scripts/reference_rgbd_width_carrier.py"),
        ("analysis", "scripts/analyze_rgbd_width_carrier.py"),
        ("statistics_audit", "scripts/audit_rgbd_width_statistics.py"),
    ):
        execute(
            root,
            label,
            [script, "--root", str(root)],
            env=env,
            times=times,
            retries=config["infrastructure_retries"],
        )
    # Primary EVAL-V4 stage: predictions sealed before query depth, then scored, analyzed.
    for stage, done in (
        ("seal", root / "raw/eval_v4/prediction_seal.json"),
        ("score", root / "raw/eval_v4/query_rows.json"),
        ("analyze", root / "eval_v4_results.json"),
    ):
        if not done.exists():
            execute(
                root,
                f"eval_v4_{stage}",
                [
                    "scripts/evaluate_rgbd_width_eval_v4.py",
                    "--root",
                    str(root),
                    "--stage",
                    stage,
                ],
                env=env,
                times=times,
                retries=config["infrastructure_retries"],
            )
    # Keep mutable subprocess output outside sealed O/D trees during finalization.
    with Path("/tmp/rgbd-width-v12-finalization.log").open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                "scripts/finalize_rgbd_width_carrier.py",
                "--root",
                str(root),
                "--docs",
                str(docs),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )
    print("ALL_PHASES_COMPLETE_FRESH_REMAINS_CLOSED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--docs", required=True)
    a = p.parse_args()
    run(a.root, a.docs)
