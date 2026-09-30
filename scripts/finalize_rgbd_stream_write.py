#!/usr/bin/env python3
"""Core B V1 independent postrun checks, preservation audit and portable archive."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_data, verify_lock
from mcss.mechanism_pilot.rgbd_stream_evaluation import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_write import CONTROLS, EXPERIMENT, POLICIES
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

METRICS = ("rgb_mse", "depth_absrel", "depth_rmse", "depth_delta1", "opacity", "coverage")


def finish(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    if root.name != EXPERIMENT or docs.name != root.name:
        raise PermissionError("Exact Core B V1 output/docs directories required")
    manifest, config = verify_lock(root)
    verify_data(root)
    old = read(root / "audit/previous_experiment_seal.json")["files"]
    for path, info in old.items():
        if sha(Path(path)) != info["sha256"]:
            raise AssertionError(f"Old experiment changed: {path}")
    tests = read(root / "tests.json")
    if tests["status"] != "PASS":
        raise AssertionError("Tests not PASS")
    for path, digest in tests.get("source_sha256", {}).items():
        if sha(Path(path)) != digest:
            raise AssertionError("Source differs from tested snapshot")
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "DEV"}
    train = {r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN"}
    for seed in config["seeds"]:
        run = root / "checkpoints" / f"write_{seed}"
        summary = read(run / "summary.json")
        logs = [json.loads(s) for s in (run / "training.jsonl").read_text().splitlines()]
        if [r["step"] for r in logs] != list(range(1, config["steps"] + 1)):
            raise AssertionError("Wrong write-rule training budget")
        if not all(r["scene_id"] in train and np.isfinite(r["loss"]) for r in logs):
            raise AssertionError("Non-TRAIN or nonfinite write-rule training")
        for name, key in (
            ("rule_step000000.pt", "initial_rule_sha256"),
            ("rule_final.pt", "final_rule_sha256"),
        ):
            if sha(run / name) != summary[key]:
                raise AssertionError("Write rule changed after training")
        for line in (run / "access.jsonl").read_text().splitlines():
            event = json.loads(line)
            if "scene_id" in event and event["scene_id"] not in train:
                raise AssertionError("Write-rule training touched a non-TRAIN scene")
    rows = read(root / "raw/dev_stream_query_results.json")
    maximum = {k: 0.0 for k in METRICS}
    for row in rows:
        path = root / row["prediction_path"]
        if sha(path) != row["prediction_file_sha256"]:
            raise AssertionError("Prediction changed")
        pred = dict(np.load(path))
        frame = next(
            f for f in records[row["scene_id"]]["frames"] if f["frame_id"] == row["query_id"]
        )
        with Image.open(frame["rgb"]) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255
        depth = np.load(frame["depth"]).squeeze()
        metrics = measurement_metrics(pred["rgb"], rgb, pred["depth"], depth, pred["opacity"])
        for key in METRICS:
            delta = abs(metrics[key] - row[key])
            maximum[key] = max(maximum[key], delta)
            if delta > 1e-6:
                raise AssertionError(f"Metric mismatch {key}: {delta}")
    names = (*POLICIES, *CONTROLS)
    for seed in config["seeds"]:
        out = root / "raw/dev_stream" / f"seed_{seed}"
        audit = read(out / "integrity.json")
        if sha(out / "states.pt") != audit["states_file_sha256"]:
            raise AssertionError("Sealed states file changed")
        states = torch.load(out / "states.pt", map_location="cpu", weights_only=False)
        index = read(out / "state_hashes.json")
        expected = {f"{s}/{role}/{n}" for s in records for role in ("A", "B") for n in names}
        if set(states) != expected or set(index) != expected:
            raise AssertionError("Incomplete sealed state population")
        for key, state in states.items():
            if hash_scene_state(state) != index[key]["state_hash"]:
                raise AssertionError("State tensor hash mismatch")
        access = read(out / "GT_access.json")
        marker = next(
            i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED"
        )
        preseal_depth_audit(access[:marker], records)
    write_json(
        root / "audit/postrun_validation.json",
        {
            "status": "PASS",
            "query_rows": len(rows),
            "old_files_unchanged": len(old),
            "max_metric_errors": maximum,
            "fresh_qualification_opened": False,
            "final_holdout_touched": False,
        },
    )
    write_json(
        root / "integrity.json",
        {
            "status": "PASS",
            "old_files_unchanged": len(old),
            "metrics_recomputed": len(rows),
            "slow_carrier_updated": False,
            "test_time_input": "RGB+DEPTH+CAMERA of warmup and stream frames only",
            "fresh_qualification_opened": False,
            "final_holdout_touched": False,
        },
    )
    subprocess.run(
        [sys.executable, "scripts/report_rgbd_stream_write.py", "--root", str(root)], check=True
    )
    (root / "audit/final_git_status.txt").write_bytes(
        subprocess.check_output(["git", "status", "--short"])
    )
    source = sorted(Path("src/mcss").rglob("*.py")) + sorted(
        Path("scripts").glob("*rgbd_stream_write*.py")
    )
    source += sorted(Path("tests").glob("test_rgbd_stream_write*.py"))
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for p in source:
            archive.add(p, arcname=str(p))
    write_json(root / "source_manifest.json", {str(p): sha(p) for p in source})
    docs.mkdir(parents=True, exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in (".json", ".md", ".txt", ".gz"):
            shutil.copyfile(p, docs / p.name)
    for name in ("figures", "audit"):
        if (root / name).exists():
            shutil.copytree(root / name, docs / name, dirs_exist_ok=True)
    print("CORE_B_V1_FINAL_AUDIT_PASS", len(rows), len(old), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()
    finish(args.root, args.docs)
