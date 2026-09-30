#!/usr/bin/env python3
"""Execute the frozen Core B V1 run on CPU: train write rules, evaluate, analyze, finalize."""

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


def execute(root, label, args, env, times):
    log = root / "audit" / f"{label}.log"
    print(label, "START", flush=True)
    tick = time.perf_counter()
    with log.open("x") as handle:
        process = subprocess.run(
            [sys.executable, *args], stdout=handle, stderr=subprocess.STDOUT, env=env
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
                    "scripts/train_rgbd_stream_write.py",
                    "--root",
                    str(root),
                    "--seed",
                    str(seed),
                    "--device",
                    "cpu",
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
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "",
        **config["thread_environment"],
    }
    times = {}
    started = time.perf_counter()
    train_all(root, config, env, times)
    for label, script in (
        ("dev_evaluation", "scripts/evaluate_rgbd_stream_write.py"),
        ("analysis", "scripts/analyze_rgbd_stream_write.py"),
    ):
        execute(root, label, [script, "--root", str(root)], env, times)
    times["total_before_finalize"] = time.perf_counter() - started
    (root / "audit/phase_wall_times.json").write_text(json.dumps(times, indent=2) + "\n")
    with Path("/tmp/rgbd-stream-write-v1-finalization.log").open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                "scripts/finalize_rgbd_stream_write.py",
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
    print("CORE_B_V1_COMPLETE_FRESH_REMAINS_CLOSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()
    run(args.root, args.docs)
