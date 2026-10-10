#!/usr/bin/env python3
"""Execute the frozen Core B V3 run on GPU 4: train write rules, evaluate, analyze, finalize.

Launched through ~/cv_tools/cv_gpu_run.sh, which picks GPU 4 only when it is idle and sets
CUDA_VISIBLE_DEVICES=4; children inherit it. Training and DEV evaluation run on cuda.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.small_training import sha

# Every child runs through the hash-locked path-mapping launcher (records store
# zonghan_o's /home/zonghan/... paths).
LAUNCHER = str(Path.home() / "cv_tools/cv_host_run.py")


def execute(root, label, args, env, times):
    log = root / "audit" / f"{label}.log"
    print(label, "START", flush=True)
    tick = time.perf_counter()
    with log.open("x") as handle:
        process = subprocess.run(
            [sys.executable, LAUNCHER, *args], stdout=handle, stderr=subprocess.STDOUT, env=env
        )
    times[label] = time.perf_counter() - tick
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args)
    print(label, "PASS", round(times[label], 1), flush=True)


def train_all(root, config, env, times):
    pending = [
        seed
        for seed in config["seeds"]
        if not (root / "checkpoints" / f"write_{seed}" / "summary.json").exists()
    ]
    with concurrent.futures.ThreadPoolExecutor(config["parallel_workers"]) as pool:
        futures = {
            pool.submit(
                execute,
                root,
                f"train_write_{seed}",
                [
                    "scripts/train_rgbd_stream_write_v3.py",
                    "--root",
                    str(root),
                    "--seed",
                    str(seed),
                    "--device",
                    "cuda",
                ],
                env,
                times,
            ): seed
            for seed in pending
        }
        concurrent.futures.wait(futures)
    failures = {seed: repr(f.exception()) for f, seed in futures.items() if f.exception()}
    if failures:
        raise RuntimeError(f"Write-rule training failed; experiment stops: {failures}")


def run(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    _, config = verify_lock(root)
    if read(root / "tests.json")["status"] != "PASS":
        raise PermissionError("Full suite must PASS before training")
    if not os.sched_getaffinity(0) <= set(config["cpu_affinity"]):
        raise PermissionError("Run must be pinned to the frozen CPU cores (taskset)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu"]:
        raise PermissionError("Run must hold the frozen GPU through cv_gpu_run.sh")
    if sha(Path(LAUNCHER)) != config["host_launcher"]["sha256"]:
        raise PermissionError("The path-mapping launcher changed after the freeze")
    if not os.environ.get("CV_ROOT"):
        raise PermissionError("CV_ROOT (the mirror root) must be set")
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        **config["thread_environment"],
    }
    times = {}
    started = time.perf_counter()
    train_all(root, config, env, times)
    for label, script, extra in (
        ("dev_evaluation", "scripts/evaluate_rgbd_stream_write_v3.py", ["--device", "cuda"]),
        ("analysis", "scripts/analyze_rgbd_stream_write_v3.py", []),
    ):
        execute(root, label, [script, "--root", str(root), *extra], env, times)
    times["total_before_finalize"] = time.perf_counter() - started
    (root / "audit/phase_wall_times.json").write_text(json.dumps(times, indent=2) + "\n")
    with Path("/tmp/rgbd-stream-write-v3-finalization.log").open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                LAUNCHER,
                "scripts/finalize_rgbd_stream_write_v3.py",
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
    print("CORE_B_V3_COMPLETE_FRESH_REMAINS_CLOSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()
    run(args.root, args.docs)
