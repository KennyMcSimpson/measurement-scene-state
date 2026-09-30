#!/usr/bin/env python3
"""Execute the frozen FRESH-V1 qualification on CPU only: evaluate, reference, analyze, report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def run(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    env = {**os.environ, **THREADS, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": ""}
    stages = [
        (
            "fresh_evaluation",
            [
                "scripts/evaluate_rgbd_fresh_qualification.py",
                "--root",
                str(root),
                "--device",
                "cpu",
            ],
        ),
        (
            "geometry_free_reference",
            ["scripts/reference_rgbd_fresh_qualification.py", "--root", str(root)],
        ),
        ("analysis", ["scripts/analyze_rgbd_fresh_qualification.py", "--root", str(root)]),
        (
            "report",
            [
                "scripts/report_rgbd_fresh_qualification.py",
                "--root",
                str(root),
                "--docs",
                str(docs),
            ],
        ),
    ]
    times = {}
    for label, command in stages:
        log = root / "audit" / f"{label}.log"
        print(label, "START", flush=True)
        tick = time.perf_counter()
        with log.open("x") as handle:
            process = subprocess.run(
                [sys.executable, *command], stdout=handle, stderr=subprocess.STDOUT, env=env
            )
        times[label] = time.perf_counter() - tick
        if process.returncode:
            raise SystemExit(f"{label} failed with {process.returncode}; see {log}")
        print(label, "PASS", round(times[label], 1), flush=True)
    (root / "audit" / "phase_wall_times.json").write_text(json.dumps(times, indent=2) + "\n")
    print("FRESH_QUALIFICATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()
    run(args.root, args.docs)
