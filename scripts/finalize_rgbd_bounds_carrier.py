#!/usr/bin/env python3
"""V7 independent postrun checks, preservation audit, and portable numeric archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.geometry_carrier_experiment import (
    PRIMARY_PAIR,
    read,
    verify_data,
    verify_lock,
)
from mcss.mechanism_pilot.rgbd_bounds_carrier import VARIANT_SPECS
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.statistics import measurement_metrics

METRICS = (
    "rgb_mse",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
)


def finish(root, docs):
    root, docs = Path(root).resolve(), Path(docs).resolve()
    if root.name != "EXP-3D-RGBD-DEPTH-BOUNDS-V7" or docs.name != root.name:
        raise PermissionError("Exact experiment paths required")
    if root == docs or root in docs.parents or docs in root.parents:
        raise PermissionError("Disjoint output/archive directories required")
    manifest, config = verify_lock(root)
    verify_data(root)
    evaluated = read(root / "evaluated_variants.json")["variants"]
    if tuple(config["primary_pair"]) != PRIMARY_PAIR or not set(PRIMARY_PAIR) <= set(evaluated):
        raise AssertionError("Primary C0/C1 must be evaluated under the frozen pair")
    old = read(root / "audit/previous_experiment_seal.json")["files"]
    for path, info in old.items():
        if sha(Path(path)) != info["sha256"]:
            raise AssertionError(f"Old experiment changed: {path}")
    tests = read(root / "tests.json")
    if tests["status"] != "PASS" or tests.get("failed", 0) or tests.get("exit_code", 0):
        raise AssertionError("Tests not PASS")
    if read(root / "audit/statistics_reproduction_audit.json")["status"] != "PASS":
        raise AssertionError("Statistics reproduction not PASS")
    for path, digest in tests.get("source_sha256", {}).items():
        if sha(Path(path)) != digest:
            raise AssertionError("Source differs from tested snapshot")
    reproduction = read(root / "audit/statistics_reproduction_audit.json")
    for section in ("reproduction_inputs_sha256", "reproduction_source_sha256"):
        for path, digest in reproduction[section].items():
            if sha(Path(path)) != digest:
                raise AssertionError("Statistics audit inputs/source changed after PASS")
    for name, row in reproduction["comparisons"].items():
        if sha(root / name) != row["original_decoded_sha256"]:
            raise AssertionError("Scientific output changed after raw reproduction")
    models = []
    for variant in evaluated:
        for seed in config["seeds"]:
            run = root / "checkpoints" / f"{variant}_{seed}"
            summary = read(run / "summary.json")
            logs = [json.loads(s) for s in (run / "training.jsonl").read_text().splitlines()]
            if len(logs) != config["steps"] or [r["step"] for r in logs] != list(
                range(1, config["steps"] + 1)
            ):
                raise AssertionError("Wrong training budget/steps")
            train = {r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN"}
            if not all(r["scene_id"] in train and np.isfinite(r["loss"]) for r in logs):
                raise AssertionError("Non-TRAIN or nonfinite training")
            selection = summary["selection"]
            if sha(Path(selection["checkpoint"])) != selection["checkpoint_sha256"]:
                raise AssertionError("Selected weights changed")
            models.append(summary)
            for checkpoint in config["checkpoint_steps"]:
                access = read(run / f"dev_{checkpoint:06d}" / "GT_access.json")
                marker = next(
                    i for i, e in enumerate(access) if e.get("event") == "ALL_DEV_STATES_SEALED"
                )
                # RGB-D track: before sealing, depth only through the context-only loader,
                # only for context frames of that scene, never for a primary query.
                by_scene = {r["scene_id"]: r for r in manifest["scenes"]}
                for event in access[:marker]:
                    if "depth" not in event.get("channels", []):
                        continue
                    roles = by_scene[event["scene_id"]]["roles"]
                    context = set(roles["context_a"]) | set(roles["context_b"])
                    if (
                        event.get("purpose") != "CONTEXT_ONLY_RGBD"
                        or event.get("frame_id") not in context
                        or event.get("frame_id") in roles["primary_query"]
                    ):
                        raise AssertionError("Non-context depth accessed before state sealing")
    for seed in config["seeds"]:
        pair = [s for s in models if s["seed"] == seed]
        if len({s["initial_parameter_hash"] for s in pair}) != 1 or any(
            len(
                {
                    s["data_ray_stream_sha256"]
                    for s in pair
                    if VARIANT_SPECS[s["variant"]]["train"] == train_set
                }
            )
            > 1
            for train_set in ("TRAIN24", "TRAIN72")
        ):
            raise AssertionError("Matched init/ray stream mismatch")
    rows = read(root / "raw/dev_query_results.json")
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "DEV"}
    maximum = {k: 0.0 for k in METRICS}
    gt_cache = {}
    pred_cache = {}
    access = []
    from PIL import Image

    for row in rows:
        if row["scene_id"] not in records:
            raise PermissionError("Result outside DEV")
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
        gt_cache[row["scene_id"], row["query_id"]] = depth.astype(np.float64)
        if row["method"] in ("direct", "anchor"):
            pred_cache[
                row["method"],
                row["variant"],
                row["seed"],
                row["scene_id"],
                row["role"],
                row["query_id"],
            ] = pred
        metrics = measurement_metrics(pred["rgb"], rgb, pred["depth"], depth, pred["opacity"])
        for key in METRICS:
            delta = abs(metrics[key] - row[key])
            maximum[key] = max(maximum[key], delta)
            if delta > 1e-6:
                raise AssertionError(f"Metric mismatch {key}: {delta}")
        access.append(
            {
                "scene_id": row["scene_id"],
                "query_id": row["query_id"],
                "purpose": "POSTRUN_METRIC_RECOMPUTATION",
                "channels": ["RGB", "depth"],
            }
        )
    for row in read(root / "raw/dev_matched_results.json"):
        sid, qid, role, seed = (row[k] for k in ("scene_id", "query_id", "role", "seed"))
        gt = gt_cache[sid, qid]
        common = np.isfinite(gt) & (gt > 0)
        for variant in PRIMARY_PAIR:
            common &= pred_cache["direct", variant, seed, sid, role, qid]["opacity"] > 1e-6
        if (
            int(common.sum()) != row["common_valid_count"]
            or hashlib.sha256(common.tobytes()).hexdigest() != row["common_mask_hash"]
        ):
            raise AssertionError("Common opacity mask mismatch")
        pred = pred_cache["direct", row["variant"], seed, sid, role, qid]
        for name, depth in (
            ("depth_absrel", pred["depth"]),
            (
                "normalized_depth_absrel",
                pred["depth"].astype(np.float64) / np.maximum(pred["opacity"], 1e-6),
            ),
        ):
            value = (
                float(np.mean(np.abs(depth[common] - gt[common]) / gt[common]))
                if common.any()
                else None
            )
            if (value is None) != (row[name] is None) or (
                value is not None and abs(value - row[name]) > 1e-6
            ):
                raise AssertionError("Matched / normalized depth metric mismatch")
    for row in read(root / "raw/dev_region_results.json"):
        sid, qid, role, seed = (row[k] for k in ("scene_id", "query_id", "role", "seed"))
        gt = gt_cache[sid, qid]
        arrays = np.load(root / "raw/observability" / f"{sid}_{role}_{qid}.npz")
        valid = np.isfinite(gt) & (gt > 0)
        mask = valid & (arrays["obs_class"] == ("OBS0", "OBS1", "OBS2PLUS").index(row["region"]))
        pred = pred_cache["direct", row["variant"], seed, sid, role, qid]["depth"]
        total = float(np.sum(np.abs(pred[mask] - gt[mask]) / gt[mask]))
        if int(mask.sum()) != row["pixel_count"] or abs(total - row["absrel_sum"]) > 1e-5:
            raise AssertionError("Regional error/count mismatch")
    static_rows = read(root / "raw/dev_static_matched_results.json")
    expected_static = (
        len(evaluated)
        * len(config["seeds"])
        * sum(2 * len(r["roles"]["primary_query"]) for r in records.values())
    )
    if len(static_rows) != expected_static:
        raise AssertionError("Incomplete static anchor/direct rows")
    for row in static_rows:
        sid, qid, role, seed = (row[k] for k in ("scene_id", "query_id", "role", "seed"))
        gt = gt_cache[sid, qid]
        direct_pred = pred_cache["direct", row["variant"], seed, sid, role, qid]
        anchor_pred = pred_cache["anchor", row["variant"], seed, sid, role, qid]
        shared = np.isfinite(gt) & (gt > 0)
        shared &= (direct_pred["opacity"] > 1e-6) & (anchor_pred["opacity"] > 1e-6)
        if (
            int(shared.sum()) != row["common_valid_count"]
            or hashlib.sha256(shared.tobytes()).hexdigest() != row["common_mask_hash"]
        ):
            raise AssertionError("Static common opacity mask mismatch")
        for side, pred in (("anchor", anchor_pred), ("direct", direct_pred)):
            for name, depth in (
                ("depth_absrel", pred["depth"]),
                (
                    "normalized_depth_absrel",
                    pred["depth"].astype(np.float64) / np.maximum(pred["opacity"], 1e-6),
                ),
            ):
                value = (
                    float(np.mean(np.abs(depth[shared] - gt[shared]) / gt[shared]))
                    if shared.any()
                    else None
                )
                saved = row[f"{side}_{name}"]
                if (value is None) != (saved is None) or (
                    value is not None and abs(value - saved) > 1e-6
                ):
                    raise AssertionError("Static anchor/direct metric mismatch")
    for variant in evaluated:
        for seed in config["seeds"]:
            out = root / "raw/selected_dev" / f"{variant}_{seed}"
            audit = read(out / "integrity.json")
            if sha(out / "states.pt") != audit["states_file_sha256"]:
                raise AssertionError("Sealed states file changed")
            states = torch.load(out / "states.pt", map_location="cpu", weights_only=False)
            indices = read(out / "state_hashes.json")
            expected = {
                f"{sid}/{role}" for sid in records for role in ("A", "B", "anchor_A", "anchor_B")
            }
            if set(states) != expected or set(indices) != expected:
                raise AssertionError("Incomplete sealed state population")
            for key, state in states.items():
                if hash_scene_state(state) != indices[key]["state_hash"]:
                    raise AssertionError("State tensor hash mismatch")
    write_json(root / "audit/postrun_GT_access.json", access)
    write_json(
        root / "audit/postrun_validation.json",
        {
            "status": "PASS",
            "query_rows": len(rows),
            "models": len(models),
            "old_files_unchanged": len(old),
            "max_metric_errors": maximum,
            "data_hashes_verified": True,
            "matched_seeds": len(config["seeds"]),
            "static_rows_recomputed": len(static_rows),
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
            "test_time_depth_used": True,
            "test_time_depth_scope": "context frames only; query depth only after seal",
            "fresh_qualification_opened": False,
            "final_holdout_touched": False,
            "dynamic_ttt_run": False,
        },
    )
    subprocess.run(
        [
            ".venv/bin/python",
            "scripts/report_rgbd_bounds_carrier.py",
            "--root",
            str(root),
            "--docs",
            str(docs),
        ],
        check=True,
    )
    # Exact patch includes new source/test files, preserving the existing shared worktree.
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    untracked = (
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "src", "scripts", "tests"]
        )
        .decode()
        .splitlines()
    )
    for name in untracked:
        diff = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", name], capture_output=True
        )
        if diff.returncode not in (0, 1):
            raise RuntimeError("Could not capture untracked code")
        patch += diff.stdout
    (root / "dirty.patch").write_bytes(patch)
    (root / "audit/final_git_status.txt").write_bytes(
        subprocess.check_output(["git", "status", "--short"])
    )
    source = sorted(Path("src/mcss").rglob("*.py")) + sorted(
        Path("scripts").glob("*rgbd_bounds*.py")
    )
    source += sorted(Path("tests").glob("test_rgbd_*bounds*.py"))
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for p in source:
            archive.add(p, arcname=str(p))
    write_json(root / "source_manifest.json", {str(p): sha(p) for p in source})
    docs.mkdir(parents=True, exist_ok=True)
    for p in root.iterdir():
        if (
            p.is_file()
            and p.name != "artifact_manifest.json"
            and p.suffix in (".json", ".md", ".txt", ".sh", ".patch", ".gz")
        ):
            shutil.copyfile(p, docs / p.name)
    (docs / "artifact_manifest.json").unlink(missing_ok=True)
    for name in ("audit", "figures"):
        for p in (root / name).rglob("*"):
            if p.is_file() and p.suffix not in (".pt", ".npz", ".npy", ".hdf5"):
                dest = docs / p.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
    # Numeric raw (including controls/selection) is portable; no image/weight duplication.
    for p in (root / "raw").rglob("*.json"):
        dest = docs / p.relative_to(root)
        dest = dest.with_suffix(dest.suffix + ".gz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
                z.write(p.read_bytes())
    for p in (root / "checkpoints").rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl"):
            dest = docs / p.relative_to(root)
            dest = dest.with_suffix(dest.suffix + ".gz")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as handle:
                with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as z:
                    z.write(p.read_bytes())
    checkpoint_index = {
        str(p.relative_to(root)): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in (root / "checkpoints").rglob("*.pt")
    }
    write_json(root / "checkpoints/index.json", checkpoint_index)
    write_json(docs / "checkpoint_index.json", checkpoint_index)
    for directory, name in ((root, "artifact_manifest.json"), (docs, "report_manifest.json")):
        values = {
            str(p.relative_to(directory)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != name
        }
        write_json(directory / name, values)
    print("FINAL_AUDIT_AND_ARCHIVE_PASS", len(rows), len(old))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--docs", required=True)
    a = p.parse_args()
    finish(a.root, a.docs)
