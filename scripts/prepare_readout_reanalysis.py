#!/usr/bin/env python3
"""Freeze the post-hoc readout/reference re-analysis before this experiment reads DEV GT."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.readout_reanalysis import (
    EXPERIMENT,
    METHODS,
    OPACITY_FLOOR,
    PRIMARY_REFERENCE,
    READOUTS,
    REFERENCES,
    SOURCES,
    VARIANTS,
    reference_values,
)
from mcss.mechanism_pilot.small_training import sha, write_json

FINAL_MARKERS = ("source_manifest.json", "dirty.patch")
SOURCE_FILES = (
    "src/mcss/mechanism_pilot/readout_reanalysis.py",
    "scripts/prepare_readout_reanalysis.py",
    "scripts/run_readout_reanalysis.py",
    "tests/test_readout_reanalysis.py",
    "src/mcss/mechanism_pilot/direct_capacity_statistics.py",
    "src/mcss/mechanism_pilot/geometry_carrier_statistics.py",
    "src/mcss/mechanism_pilot/statistics.py",
)


def dev_roster(manifest):
    return {
        r["scene_id"]: {
            "primary_query": list(r["roles"]["primary_query"]),
            "depth": {
                str(f["frame_id"]): f["depth"]
                for f in r["frames"]
                if f["frame_id"] in r["roles"]["primary_query"]
            },
        }
        for r in manifest["scenes"]
        if r["split"] == "DEV"
    }


def train_roster(manifest):
    return sorted(r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=f"outputs/{EXPERIMENT}")
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(root)
    manifests, inputs, seeds = {}, {}, {}
    for key, spec in SOURCES.items():
        base = Path(spec["root"])
        for marker in FINAL_MARKERS:
            if not (base / marker).is_file():
                raise PermissionError(f"{key} is not finalized: {marker} missing")
        path = base / "scene_split.json"
        manifests[key] = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        contract = base / "training_contract.json"
        inputs[str(contract)] = sha(contract)
        seeds[key] = list(json.loads(contract.read_text())["seeds"])
    rosters = [dev_roster(m) for m in manifests.values()]
    if any(r != rosters[0] for r in rosters) or len(rosters[0]) != 8:
        raise ValueError("V2-V4 must share one identical 8-scene DEV primary-query roster")
    trains = [train_roster(m) for m in manifests.values()]
    if any(t != trains[0] for t in trains) or len(trains[0]) != 24:
        raise ValueError("V2-V4 must share one identical 24-scene TRAIN roster")
    if any(s != seeds["V2"] for s in seeds.values()):
        raise ValueError("V2-V4 must share the frozen seeds")
    predictions = []
    for key, spec in SOURCES.items():
        selected = Path(spec["root"]) / "raw" / "selected_dev"
        for variant in VARIANTS:
            for seed in seeds[key]:
                directory = selected / f"{variant}_{seed}"
                results = directory / "query_results.json"
                inputs[str(results)] = sha(results)
                for row in json.loads(results.read_text()):
                    if row["method"] not in METHODS:
                        continue
                    path = directory / row["prediction_path"]
                    digest = sha(path)
                    if digest != row["prediction_file_sha256"]:
                        raise PermissionError(f"Sealed prediction changed: {path}")
                    predictions.append(
                        {
                            "source": key,
                            "variant": variant,
                            "seed": seed,
                            "scene_id": row["scene_id"],
                            "role": row["role"],
                            "query_id": row["query_id"],
                            "method": row["method"],
                            "step": row["step"],
                            "path": str(path),
                            "sha256": digest,
                            "sealed_depth_absrel": row["depth_absrel"],
                            "sealed_depth_delta1": row["depth_delta1"],
                            "sealed_depth_valid_count": row["depth_valid_count"],
                        }
                    )
    # Geometry-free constants from TRAIN supervision targets only (primary-query GT).
    manifest = manifests["V4"]
    train_files, train_depths = {}, []
    for record in sorted(
        (r for r in manifest["scenes"] if r["split"] == "TRAIN"), key=lambda r: r["scene_id"]
    ):
        frames = {f["frame_id"]: f for f in record["frames"]}
        for fid in record["roles"]["primary_query"]:
            path = Path(frames[fid]["depth"])
            train_files[str(path)] = sha(path)
            train_depths.append(np.load(path).squeeze().astype(np.float32))
    references = reference_values(train_depths)
    source = {name: sha(Path(name)) for name in SOURCE_FILES}
    lock = {
        "experiment": EXPERIMENT,
        "status": "FROZEN_BEFORE_DEV_GT_ANALYSIS",
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "post-hoc secondary re-analysis of sealed predictions; no training",
        "dev_gt_read_by_this_experiment_before_lock": False,
        "frozen_statuses_unchanged": True,
        "sources": {k: {**v, "seeds": seeds[k]} for k, v in SOURCES.items()},
        "prediction_count": len(predictions),
        "predictions": predictions,
        "input_sha256": inputs,
        "train_reference_files_sha256": train_files,
        "references": references,
        "analysis": {
            "readouts": {
                "RAW": "sealed renderer depth sum_i w_i t_i (the frozen metric input)",
                "OPACITY_NORMALIZED": f"RAW / max(opacity, {OPACITY_FLOOR})",
                "MEDIAN_SCALED": "RAW * median(GT) / median(RAW) per query view on GT-valid "
                "pixels; uses GT scale, a shape-only diagnostic",
            },
            "readout_order": list(READOUTS),
            "metrics": "measurement_metrics depth_absrel and depth_delta1 on GT-valid pixels",
            "references": {
                "REF_TRAIN_ABSREL_OPTIMAL": "1/t-weighted median of pooled TRAIN "
                "primary-query GT: the AbsRel-optimal constant",
                "REF_TRAIN_MEDIAN": "median of pooled TRAIN primary-query GT",
                "under_MEDIAN_SCALED": "both become the per-view GT-median constant",
            },
            "reference_order": list(REFERENCES),
            "primary_reference": PRIMARY_REFERENCE,
            "aggregation": "frozen V2 _macro: seeds per scene/role/query, then queries, "
            "roles, scenes",
            "bootstrap": {"unit": "scene", "draws": 10000, "seed": 20260929, "CI": "pct95"},
            "labels": {
                "ABOVE": "frozen V2 evidence rule: mean>0, CIlo>0, >=75% scenes not worse",
                "BELOW": "mirror rule: mean<0, CIhi<0, >=75% scenes not better",
                "NOT_DISTINGUISHABLE": "otherwise",
            },
            "summary_status": {
                "NO_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE": "no V2-V4 carrier variant ABOVE "
                "its primary reference on RAW or OPACITY_NORMALIZED AbsRel/delta1",
                "ABOVE_REFERENCE_ONLY_AFTER_OPACITY_NORMALIZATION": "some ABOVE only under "
                "OPACITY_NORMALIZED",
                "SOME_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE": "some ABOVE under RAW",
            },
            "integrity": "RAW recomputation must reproduce every sealed depth_absrel within "
            "1e-5 and depth_delta1 within 1/valid_count, else abort",
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": source,
    }
    root.mkdir(parents=True)
    for name in ("raw", "audit"):
        (root / name).mkdir()
    write_json(root / "preregistration.json", lock)
    print(json.dumps({k: lock[k] for k in ("status", "prediction_count", "references")}))


if __name__ == "__main__":
    main()
