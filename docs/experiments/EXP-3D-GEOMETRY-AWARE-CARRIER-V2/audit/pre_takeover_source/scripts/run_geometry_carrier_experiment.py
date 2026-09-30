#!/usr/bin/env python3
"""Execute the frozen six-run experiment and audits, never opening fresh data."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.small_training import write_json


def run(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    _, config = verify_lock(root)
    if read(root / "tests.json")["status"] != "PASS":
        raise PermissionError("Full suite must PASS before training")
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    times = {}

    def execute(label, args):
        print(label, "START", flush=True)
        tick = time.perf_counter()
        with (root / "audit" / f"{label}.log").open("x") as handle:
            subprocess.run(
                [sys.executable, *args],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                env=env,
            )
        times[label] = time.perf_counter() - tick
        print(label, "PASS", times[label], flush=True)

    start = time.perf_counter()
    for seed in config["seeds"]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    execute,
                    f"train_{variant}_{seed}",
                    [
                        "scripts/train_geometry_carrier.py",
                        "--root",
                        str(root),
                        "--variant",
                        variant,
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda",
                    ],
                )
                for variant in config["variants"]
            ]
            for f in futures:
                f.result()
    execute("selected_dev", ["scripts/evaluate_geometry_carrier.py", "--root", str(root)])
    write_json(
        root / "audit/phase_wall_times.json",
        {
            "phases": times,
            "wall_seconds": time.perf_counter() - start,
            "parallel_workers": 2,
            "shared_gpu": True,
        },
    )
    execute("analysis", ["scripts/analyze_geometry_carrier.py", "--root", str(root)])
    execute(
        "statistics_audit", ["scripts/audit_geometry_carrier_statistics.py", "--root", str(root)]
    )
    # Keep mutable subprocess output outside sealed O/D trees during finalization.
    with Path("/tmp/geometry-carrier-v2-finalization.log").open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                "scripts/finalize_geometry_carrier.py",
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
