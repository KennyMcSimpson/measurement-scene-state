"""Finite corrected-camera carrier training and train-only policy refinement run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_dynamic_policy_refinement import sha256, write_json

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.relative_to(PROJECT / "outputs")
    if (root / "pipeline_status.json").exists() or (root / "training").exists():
        raise FileExistsError("refusing to restart or overwrite a previously launched pipeline")
    train_index = root / "dataset/episodes/episode_index_train.json"
    dev_index = root / "dataset/episodes/episode_index_dev.json"
    inventory = json.loads((root / "dataset/inventory/run_inventory.json").read_text())
    if inventory.get("rigid_cameras_only") is not True:
        raise ValueError("pipeline requires an explicitly recorded rigid-camera filter")
    config = root / "training_config.json"
    status = {
        "schema": "mcss.dynamic.rigid_pipeline.v1", "status": "running", "stage": None,
        "pid": os.getpid(), "stages": [], "started_at": datetime.now(UTC).isoformat(),
        "train_index_sha256": sha256(train_index), "dev_index_sha256": sha256(dev_index),
        "training_config_sha256": sha256(config),
        "scope": "fresh carrier training, same-protocol development, policy recollection/refit",
        "external_benchmark_status": "not_started",
    }
    final_carrier = root / "training/phase_b_final.pt"
    common_eval = [
        "--index", str(dev_index), "--device", "cuda", "--render-samples", "48",
        "--ray-chunk-size", "2048",
    ]
    stages = [
        ("train_carrier", "train_dynamic_experiment.py", [
            "--episode-index", str(train_index), "--output-dir", str(root / "training"),
            "--device", "cuda", "--config-json", str(config),
        ]),
        ("evaluate_old_carrier_on_corrected_protocol", "evaluate_dynamic_checkpoint.py", [
            *common_eval, "--checkpoint",
            str(PROJECT / "outputs/dynamic_full_method_20260920/training/phase_b_final.pt"),
            "--output", str(root / "evaluation_dev_old_carrier"),
            "--policies", "OFF", "ALL",
        ]),
        ("evaluate_corrected_carrier", "evaluate_dynamic_checkpoint.py", [
            *common_eval, "--checkpoint", str(final_carrier),
            "--output", str(root / "evaluation_dev_fixed"),
            "--policies", "OFF", "FUSE", "COMPLETE", "ALL",
        ]),
        ("policy_refinement", "run_dynamic_policy_refinement.py", [
            "--train-index", str(train_index), "--dev-index", str(dev_index),
            "--checkpoint", str(final_carrier), "--output", str(root / "policy_refinement"),
        ]),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    try:
        for name, script, arguments in stages:
            command = [sys.executable, str(PROJECT / "scripts" / script), *arguments]
            entry = {"name": name, "started_at": datetime.now(UTC).isoformat(), "command": command}
            status["stage"] = name
            status["stages"].append(entry)
            write_json(root / "pipeline_status.json", status)
            print(f"Starting {name}", flush=True)
            with (root / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout:
                with (root / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
                    result = subprocess.run(
                        command, cwd=PROJECT, env=env, stdout=stdout, stderr=stderr, check=False,
                    )
            entry["returncode"] = result.returncode
            entry["finished_at"] = datetime.now(UTC).isoformat()
            if result.returncode:
                raise RuntimeError(f"stage {name} failed; see its stderr log")
            write_json(root / "pipeline_status.json", status)
        status["status"] = "complete"
        status["stage"] = "development_complete_external_benchmark_pending"
        status["final_carrier_sha256"] = sha256(final_carrier)
        status["final_policy_sha256"] = sha256(root / "policy_refinement/policy_recollected.pt")
    except Exception as error:
        status["status"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        write_json(root / "pipeline_status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
