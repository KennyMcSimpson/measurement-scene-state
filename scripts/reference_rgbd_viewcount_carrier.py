#!/usr/bin/env python3
"""V9 preregistered geometry-free reference for the sealed selected DEV predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.readout_reanalysis import (
    METHODS,
    analyze_with_sensitivity,
    score_predictions,
    score_references,
    validate_matrix,
)
from mcss.mechanism_pilot.small_training import sha, write_json

SOURCE = {
    "V9": {
        "labels": {"C0": "FIXED3", "C1": "VARIABLE3TO7", "C2": "FIXED7"},
        "primary": "VIEWCOUNT_GAIN",
    }
}


def sealed_predictions(root, variants, seeds):
    predictions = []
    for variant in variants:
        for seed in seeds:
            directory = root / "raw" / "selected_dev" / f"{variant}_{seed}"
            for row in read(directory / "query_results.json"):
                if row["method"] not in METHODS:
                    continue
                path = directory / row["prediction_path"]
                if sha(path) != row["prediction_file_sha256"]:
                    raise PermissionError(f"Sealed prediction changed: {path}")
                predictions.append(
                    {
                        "source": "V9",
                        "variant": variant,
                        "seed": seed,
                        "scene_id": row["scene_id"],
                        "role": row["role"],
                        "query_id": row["query_id"],
                        "method": row["method"],
                        "step": row["step"],
                        "path": str(path),
                        "sealed_depth_absrel": row["depth_absrel"],
                        "sealed_depth_delta1": row["depth_delta1"],
                        "sealed_depth_valid_count": row["depth_valid_count"],
                    }
                )
    return predictions


def run(root):
    root = Path(root).resolve()
    manifest, config = verify_lock(root)
    rules = read(root / "decision_rules.json")["GEOMETRY_FREE_REFERENCE"]
    variants = read(root / "evaluated_variants.json")["variants"]
    predictions = sealed_predictions(root, variants, config["seeds"])
    # Every prediction hash is verified above, after all DEV states were sealed.
    access, gt, queries = [], {}, {}
    for record in manifest["scenes"]:
        if record["split"] != "DEV":
            continue
        queries[record["scene_id"]] = list(record["roles"]["primary_query"])
        frames = {f["frame_id"]: f for f in record["frames"]}
        for fid in record["roles"]["primary_query"]:
            path = Path(frames[fid]["depth"])
            access.append(
                {
                    "scene_id": record["scene_id"],
                    "frame_id": fid,
                    "channel": "depth",
                    "purpose": "POST_SEAL_GEOMETRY_FREE_REFERENCE",
                }
            )
            data = np.load(path).squeeze().astype(np.float32)
            valid = np.isfinite(data) & (data > 0)
            gt[record["scene_id"], fid] = np.where(valid, data, 0).astype(np.float32)
    rows, worst = score_predictions(predictions, gt)
    reference_rows = score_references(rules["values"], gt, queries)
    validate_matrix(
        rows,
        reference_rows,
        config["seeds"],
        sorted(queries),
        queries,
        sources=SOURCE,
        variants=variants,
    )
    result = analyze_with_sensitivity(
        rows,
        reference_rows,
        sources=SOURCE,
        variants=variants,
        draws=rules["bootstrap"]["draws"],
        seed=rules["bootstrap"]["seed"],
    )
    result["primary_reference"] = rules["primary_reference"]
    result["reference_values"] = rules["values"]
    result["integrity"] = {
        "max_raw_absrel_deviation": worst["absrel"],
        "max_raw_delta1_pixel_deviation": worst["delta1_pixels"],
        "median_scale_undefined_views": worst["median_scale_undefined_views"],
        "prediction_rows": len(rows),
        "reference_rows": len(reference_rows),
        "variants": variants,
    }
    write_json(root / "audit" / "reference_gt_access.json", access)
    write_json(root / "raw" / "dev_reference_rows.json", reference_rows)
    write_json(root / "raw" / "dev_readout_rows.json", rows)
    write_json(root / "reference_results.json", result)
    print(result["summary"]["CARRIER_REFERENCE_STATUS"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    run(parser.parse_args().root)
