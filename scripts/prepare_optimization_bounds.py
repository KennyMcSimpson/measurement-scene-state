#!/usr/bin/env python3
"""Seal prior artifacts and prepare exact frozen camera-only bounds, without media."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from mcss.dynamic.types import hash_value
from mcss.mechanism_pilot.direct_capacity_statistics import scene_values
from mcss.mechanism_pilot.optimization_bounds_contracts import (
    CURRENT_BOUNDS,
    FrozenTrainingPrior,
    context_camera_bundle,
    frozen_gt_free_bounds,
)
from mcss.mechanism_pilot.small_training import sha, write_json

OLD = Path("outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1")
OLD_DOCS = Path("docs/experiments/EXP-3D-DIRECT-STATE-CAPACITY-V1")
CHECKPOINT = Path(
    "outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt"
)
EXPECTED_CHECKPOINT = "be7b8b6d2ef366cad245732b9ff227802db596da65082ac0e160f24541579e42"
METRICS = (
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "rgb_mse",
    "rgb_ssim",
    "opacity",
    "coverage",
)


def seal_previous(root):
    checks = []
    for base, name in ((OLD, "artifact_manifest.json"), (OLD_DOCS, "report_manifest.json")):
        expected = json.loads((base / name).read_text())
        for relative, digest in expected.items():
            path = base / relative
            if sha(path) != digest:
                raise RuntimeError(f"Previous archive manifest mismatch: {path}")
        checks.append(
            {
                "manifest": str(base / name),
                "sha256": sha(base / name),
                "verified_files": len(expected),
            }
        )
    source = json.loads((OLD / "source_manifest.json").read_text())
    for name, digest in source.items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Previous frozen source changed: {name}")
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("Frozen carrier checkpoint changed")
    files = {}
    for base in (OLD, OLD_DOCS):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files[str(path)] = {
                    "sha256": sha(path),
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
    for name, digest in source.items():
        files[name] = {
            "sha256": digest,
            "bytes": Path(name).stat().st_size,
            "mtime_ns": Path(name).stat().st_mtime_ns,
        }
    report = {
        "status": "PASS",
        "created_unix": time.time(),
        "previous_manifests": checks,
        "previous_source_manifest_sha256": sha(OLD / "source_manifest.json"),
        "checkpoint_sha256": sha(CHECKPOINT),
        "files": files,
        "query_media_decoded": False,
        "holdout_media_read": False,
        "scope": (
            "Only prior capacity output/doc artifacts and frozen source; no source datasets opened."
        ),
    }
    write_json(root / "audit/previous_experiment_seal.json", report)
    return report


def prepare(root, docs):
    root, docs = Path(root), Path(docs)
    root.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    (root / "audit").mkdir(exist_ok=True)
    seal = seal_previous(root)
    for name in ("scene_manifest.json", "train_depth_prior.json"):
        if (root / name).exists():
            raise FileExistsError(f"Refusing to overwrite locked file: {root / name}")
        shutil.copy2(OLD / name, root / name)
    manifest = json.loads((root / "scene_manifest.json").read_text())
    ids = [s["scene_id"] for s in manifest["scenes"]]
    if (
        len(ids) != 17
        or len(set(ids)) != 17
        or set(ids) & set(manifest["FINAL_HOLDOUT_PROHIBITED"])
    ):
        raise RuntimeError("Exposed scene roster invalid")
    raw_prior = json.loads((root / "train_depth_prior.json").read_text())
    prior = FrozenTrainingPrior(
        raw_prior["near_m"], raw_prior["far_m"], sha(root / "train_depth_prior.json")
    )
    previous = json.loads((OLD / "predeclared_geometry_bounds.json").read_text())["bounds"]
    bounds = {}
    for scene in manifest["scenes"]:
        for role in ("A", "B"):
            key = f"{scene['scene_id']}/{role}"
            bundle = context_camera_bundle(scene, role, manifest["image_size"])
            alternative = frozen_gt_free_bounds(bundle, prior)
            old = torch.tensor(previous[key]["bounds"], dtype=torch.float64)
            if not torch.equal(old, alternative):
                raise RuntimeError(f"GT-free bounds no longer bit-exact: {key}")
            current = torch.tensor(CURRENT_BOUNDS, dtype=torch.float64)
            bounds[key] = {
                "scene_id": scene["scene_id"],
                "role": role,
                "context_ids": list(bundle.frame_ids),
                "camera_hash": hash_value(bundle.cameras),
                "anchor_hash": hash_value(bundle.anchor_c2w),
                "prior_hash": prior.source_sha256,
                "CURRENT_BOUNDS": current.tolist(),
                "FROZEN_GT_FREE_BOUNDS": alternative.tolist(),
                "current_bounds_hash": hash_value(current),
                "bounds_hash": hash_value(alternative),
                "previous_alternative_exact": True,
            }
    write_json(
        root / "bounds_contract.json",
        {
            "status": "FROZEN_EXACT_PREVIOUS_ALTERNATIVE",
            "inputs_allowed": [
                "context_intrinsics",
                "context_extrinsics",
                "frozen_training_depth_prior",
            ],
            "inputs_forbidden": [
                "context_depth_current_scene",
                "query_camera",
                "query_depth",
                "query_rgb",
                "carrier_prediction",
                "query_score",
            ],
            "rule": (
                "all128x160 context unit rays at train q01/q99, anchor-camera coordinates, "
                "AABB5%per-side padding, minextent1m; float64 thenstateconsumerfloat32"
            ),
            "near_m": prior.near_m,
            "far_m": prior.far_m,
            "prior_hash": prior.source_sha256,
            "previous_bounds_artifact_sha256": sha(OLD / "predeclared_geometry_bounds.json"),
            "camera_hash_definition": "hash_value(tuple of float64 Cameras in fixed context order)",
            "bounds_hash_definition": "hash_value(float64[2,3] tensor)",
            "current_bounds": CURRENT_BOUNDS,
            "context_roles_unchanged": True,
            "source_sha256": sha(Path("src/mcss/mechanism_pilot/optimization_bounds_contracts.py")),
        },
    )
    write_json(
        root / "bounds_lock.json",
        {
            "status": "PASS",
            "n_scenes": 17,
            "n_contexts": 34,
            "scene_manifest_sha256": sha(root / "scene_manifest.json"),
            "prior_sha256": prior.source_sha256,
            "contract_sha256": sha(root / "bounds_contract.json"),
            "bounds": bounds,
            "current_scene_depth_used": False,
            "query_camera_or_GT_used": False,
            "all34_previous_alternatives_bit_exact": True,
        },
    )
    baseline = json.loads((OLD / "baseline_results.json").read_text())
    shutil.copy2(OLD / "baseline_results.json", root / "baseline_results.json")
    old_results = json.loads((OLD / "direct_state_results.json").read_text())
    comparisons = {}
    for name, methods in [("CARRIER", ("A", "B")), ("ANCHOR", ("anchor",)), ("PRIOR", ("prior",))]:
        rows = [dict(r, role=r["method"]) for r in baseline if r["method"] in methods]
        for metric in METRICS:
            values = scene_values(rows, metric)
            expected = old_results["frozen_carrier"][name]["metrics"][metric]["per_scene"]
            if values != expected:
                raise RuntimeError(f"Old raw baseline recomputation changed: {name}/{metric}")
            comparisons[f"{name}/{metric}"] = {
                "mean": sum(values.values()) / len(values),
                "per_scene": values,
                "exact": True,
            }
    # Reference previous direct 8/16/32 and oracle groups from numeric raw only.
    refs = {}
    context = json.loads((OLD / "raw/context_results.json").read_text())
    oracle = json.loads((OLD / "raw/oracle_results.json").read_text())
    for track, rows, block in [
        ("RGBD", context, "context_only"),
        ("QUERY_ORACLE", oracle, "diagnostic_oracle"),
    ]:
        for grid in (8, 16, 32):
            selected = [
                r
                for r in rows
                if r["track"] == track
                and r["grid"] == grid
                and r["samples"] == 64
                and r["bounds_mode"] == "CURRENT_BOUNDS"
                and r["method"] == "direct"
            ]
            values = scene_values(selected, "depth_absrel")
            key = f"{track}|g{grid}|s64|CURRENT_BOUNDS|direct"
            if values != old_results[block][key]["metrics"]["depth_absrel"]["per_scene"]:
                raise RuntimeError(f"Old reference raw changed: {key}")
            refs[key] = {
                "per_scene": values,
                "mean": sum(values.values()) / len(values),
                "exact": True,
            }
    write_json(
        root / "baseline_reproduction.json",
        {
            "status": "PASS",
            "carrier_checkpoint_sha256": EXPECTED_CHECKPOINT,
            "baseline_rows": len(baseline),
            "baseline_metrics": comparisons,
            "previous_direct_references": refs,
            "source_raw_sha256": {
                str(OLD / n): sha(OLD / n)
                for n in [
                    "baseline_results.json",
                    "raw/context_results.json",
                    "raw/oracle_results.json",
                    "direct_state_results.json",
                ]
            },
            "scope": (
                "Existing raw numeric recomputation only; no model, optimizer, "
                "query camera or media read."
            ),
            "FINAL_HOLDOUT_TOUCHED": False,
        },
    )
    return {
        "status": "PASS",
        "sealed_file_count": len(seal["files"]),
        "bounds_contexts": len(bounds),
        "scene_manifest": str(root / "scene_manifest.json"),
        "baseline_reproduction": "PASS_NUMERIC_RAW_EXACT",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--docs", required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.root, a.docs), indent=2))
