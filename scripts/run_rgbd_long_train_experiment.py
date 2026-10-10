#!/usr/bin/env python3
"""Execute the frozen V17 experiment on hit (CPU only, AVX2 kernels), never opening fresh data.

Derived from the V14 runner. Every child runs through the hash-locked path-mapping launcher
(records from zonghan_o store /home/zonghan/... paths) with the frozen thread and ISA environment.
Phases: the three training runs (one core each), the built-in V11 C1 control (stops on FAIL),
the EVAL seal of each cohort (concurrently), score, analyze and the report. A signal-terminated
run is archived and rerun from scratch within one total budget per run; any other error stops.
The process must be pinned to the frozen CPU cores (taskset -c 8-23); children inherit it.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.small_training import sha, write_json

_LOCK = threading.Lock()


def archive_attempt(root, label, attempt, returncode, archive):
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
                "kind": "native_crash",
                "returncode": returncode,
                "archived_to": str(failed),
                "moved": moved,
                "unix_time": time.time(),
                "policy": "signal-terminated: archive, rerun identical command from scratch",
            }
        )
        write_json(record, incidents)


def execute(root, label, args, *, env, launcher, times, archive=(), retries=1):
    """Run one frozen phase through the launcher; retry only signal-terminated runs."""
    root = Path(root)
    failed = root / "audit" / "failed_attempts"
    prior = len(list(failed.glob(f"{label}_attempt*"))) if failed.exists() else 0
    if prior > retries:
        raise PermissionError(f"{label}: retry budget already spent")
    for attempt in range(prior + 1, retries + 2):
        print(label, "START attempt", attempt, flush=True)
        tick = time.perf_counter()
        log = root / "audit" / f"{label}.log"
        with log.open("x") as handle:
            process = subprocess.run(
                [sys.executable, launcher, *args], stdout=handle, stderr=subprocess.STDOUT, env=env
            )
        if process.returncode == 0:
            times[label] = time.perf_counter() - tick
            print(label, "PASS", times[label], flush=True)
            return
        if process.returncode >= 0:
            raise subprocess.CalledProcessError(process.returncode, args)
        archive_attempt(root, label, attempt, process.returncode, archive)
        print(label, "NATIVE_CRASH", process.returncode, "archived", flush=True)
    raise subprocess.CalledProcessError(process.returncode, args)


def parallel(root, jobs, *, env, launcher, times, retries, workers):
    """Run {label: (args, archive)} concurrently; any failure stops the experiment."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                execute,
                root,
                label,
                args,
                env=env,
                launcher=launcher,
                times=times,
                archive=archive,
                retries=retries,
            ): label
            for label, (args, archive) in jobs.items()
        }
        concurrent.futures.wait(futures)
    failures = {futures[f]: repr(f.exception()) for f in futures if f.exception() is not None}
    if failures:
        record = root / "audit" / "phase_failures.json"
        write_json(record, {**(read(record) if record.exists() else {}), **failures})
        raise RuntimeError(f"Phase failed; experiment stops: {sorted(failures)}")


def run(root):
    root = Path(root).resolve()
    _, config = verify_lock(root)
    if read(root / "tests.json")["status"] != "PASS":
        raise PermissionError("Full suite must PASS before training")
    if not os.sched_getaffinity(0) <= set(config["cpu_affinity"]):
        raise PermissionError("Run must be pinned to the frozen CPU cores (taskset)")
    launcher = config["host_launcher"]["path"]
    if sha(Path(launcher)) != config["host_launcher"]["sha256"]:
        raise PermissionError("The path-mapping launcher changed after the freeze")
    if not os.environ.get("CV_ROOT"):
        raise PermissionError("CV_ROOT (the mirror root) must be set")
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "",
        **config["thread_environment"],
    }
    retries = config["infrastructure_retries"]
    times = {}
    start = time.perf_counter()
    training = {
        f"train_L_{seed}": (
            [
                "scripts/train_rgbd_long_train_carrier.py",
                "--root",
                str(root),
                "--variant",
                "L",
                "--seed",
                str(seed),
                "--device",
                "cpu",
            ],
            [root / "checkpoints" / f"L_{seed}"],
        )
        for seed in config["seeds"]
        if not (root / "checkpoints" / f"L_{seed}" / "summary.json").exists()
    }
    (root / "checkpoints").mkdir(exist_ok=True)
    parallel(
        root,
        training,
        env=env,
        launcher=launcher,
        times=times,
        retries=retries,
        workers=config["parallel_workers"],
    )
    if not (root / "audit/v11_prefix_control.json").exists():
        execute(
            root,
            "v11_prefix_control",
            ["scripts/check_rgbd_long_train_prefix.py", "--root", str(root)],
            env=env,
            launcher=launcher,
            times=times,
            retries=0,
        )
    if read(root / "audit/v11_prefix_control.json")["status"] != "PASS":
        raise RuntimeError("V17 does not reproduce V11 C1 for steps 1-6000; experiment stops")
    seals = {
        f"eval_seal_{name}": (
            [
                "scripts/evaluate_rgbd_long_train_eval.py",
                "--root",
                str(root),
                "--stage",
                "seal",
                "--cohort",
                name,
            ],
            [root / "raw" / "eval" / name],
        )
        for name in config["eval"]["cohorts"]
        if not (root / "raw" / "eval" / name / "prediction_seal.json").exists()
    }
    parallel(
        root,
        seals,
        env=env,
        launcher=launcher,
        times=times,
        retries=retries,
        workers=len(seals) or 1,
    )
    for stage, done in (
        ("score", root / "raw/eval/query_rows.json"),
        ("analyze", root / "eval_results.json"),
    ):
        if not done.exists():
            execute(
                root,
                f"eval_{stage}",
                ["scripts/evaluate_rgbd_long_train_eval.py", "--root", str(root), "--stage", stage],
                env=env,
                launcher=launcher,
                times=times,
                retries=retries,
            )
    execute(
        root,
        "report",
        ["scripts/report_rgbd_long_train_carrier.py", "--root", str(root)],
        env=env,
        launcher=launcher,
        times=times,
        retries=0,
    )
    wall = root / "audit/phase_wall_times.json"
    previous = read(wall) if wall.exists() else {"phases": {}, "invocations": []}
    write_json(
        wall,
        {
            "phases": {**previous["phases"], **times},
            "invocations": [*previous.get("invocations", []), time.perf_counter() - start],
            "parallel_workers": config["parallel_workers"],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    )
    print("ALL_PHASES_COMPLETE_FRESH_REMAINS_CLOSED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
