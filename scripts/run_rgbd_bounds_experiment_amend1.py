#!/usr/bin/env python3
"""V7 Amendment 1 resume: amended analysis and audit, then the frozen finalization (CPU only).

Training, selected-DEV evaluation, the state-correlation diagnostic and the geometry-free
reference already completed under the frozen runner; they are verified to exist and never rerun.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.small_training import sha, write_json

AMENDMENT_SHA256 = "491de449b5cd9f4b55994e5574e287250b69ce881524fefb83918cd30251155b"
PHASES = (
    ("analysis_amend1", "scripts/analyze_rgbd_bounds_carrier_amend1.py"),
    ("statistics_audit_amend1", "scripts/audit_rgbd_bounds_statistics_amend1.py"),
)


def archive_failed_analysis(root):
    log = root / "audit/analysis.log"
    if not log.exists():
        return
    failed = root / "audit/failed_attempts"
    failed.mkdir(parents=True, exist_ok=True)
    target = failed / "analysis_attempt1_v2_ray_hit_equality_assertion.log"
    if target.exists():
        raise FileExistsError(target)
    shutil.move(log, target)
    write_json(
        failed / "analysis_attempt1.json",
        {
            "phase": "analysis",
            "error": "ValueError: Matched variants must share frozen geometry ray-hit fractions",
            "classification": "ESTIMATOR_PREMISE_INAPPLICABLE_NOT_INFRASTRUCTURE",
            "amendment": "AMENDMENTS.md, Amendment 1",
            "log_sha256": sha(target),
        },
    )


def run(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    _, config = verify_lock(root)
    if sha(root / "AMENDMENTS.md") != AMENDMENT_SHA256:
        raise PermissionError("Amendment 1 text changed after it was frozen")
    if read(root / "tests.json")["status"] != "PASS":
        raise PermissionError("Full suite must PASS")
    if not os.sched_getaffinity(0) <= set(config["cpu_affinity"]):
        raise PermissionError("Run must be pinned to the frozen CPU cores (taskset)")
    for name in ("audit/dev_state_use.json", "reference_results.json", "state_correlation.json"):
        if not (root / name).exists():
            raise PermissionError(f"Frozen phase output missing: {name}")
    archive_failed_analysis(root)
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "",
        **config["thread_environment"],
    }
    times = {}
    for label, script in PHASES:
        tick = time.perf_counter()
        with (root / "audit" / f"{label}.log").open("x") as handle:
            subprocess.run(
                [sys.executable, script, "--root", str(root)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                env=env,
            )
        times[label] = time.perf_counter() - tick
    wall = root / "audit/phase_wall_times.json"
    previous = read(wall)
    write_json(wall, {**previous, "phases": {**previous["phases"], **times}})
    with Path("/tmp/rgbd-bounds-v7-finalization.log").open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                "scripts/finalize_rgbd_bounds_carrier.py",
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
    print("ALL_PHASES_COMPLETE_AMENDMENT1_FRESH_REMAINS_CLOSED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--docs", required=True)
    a = p.parse_args()
    run(a.root, a.docs)
