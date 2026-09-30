#!/usr/bin/env python3
"""Post-all-formal-state-seal geometry labels and saved-prediction region diagnostics."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.context_observability import audit_context_observability
from mcss.mechanism_pilot.direct_capacity_contracts import (
    CapacityEvaluator,
    ContextOnlyLoader,
    StateSealBarrier,
)
from mcss.mechanism_pilot.observability_supervision_evaluation import ensure_phase
from mcss.mechanism_pilot.small_training import sha, write_json

CAPACITY = Path("outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1")
PREVIOUS = Path("outputs/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1")


def read(path):
    return json.loads(Path(path).read_text())


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def region_masks(arrays):
    valid, obs = arrays["query_valid"], arrays["obs_class"]
    angle = arrays["triangulation_angle_bin"]
    return {
        "OVERALL": valid,
        "OBS0": valid & (obs == 0),
        "OBS1": valid & (obs == 1),
        "OBS2PLUS": valid & (obs == 2),
        "OBS01": valid & (obs < 2),
        **{
            name: valid & (angle == i)
            for i, name in enumerate(("ANGLE_0_5", "ANGLE_5_15", "ANGLE_15_30", "ANGLE_GT30"))
        },
        "OCCLUDED_ANY": valid & arrays["occluded_by_context_surface"].any(0),
        "CONFLICT_ANY": valid & arrays["depth_conflict_front"].any(0),
    }


def region_rows(row, prediction, arrays):
    gt = arrays["query_gt_depth"]
    predicted = np.asarray(prediction).squeeze()
    if predicted.shape != gt.shape or not np.isfinite(predicted).all():
        raise ValueError("Finite depth prediction with full query dimensions required")
    valid = arrays["query_valid"]
    error = np.zeros(gt.shape, dtype=np.float64)
    error[valid] = np.abs(predicted[valid].astype(np.float64) - gt[valid]) / gt[valid]
    output = []
    for region, mask in region_masks(arrays).items():
        count = int(mask.sum())
        total = float(error[mask].sum())
        output.append(
            {
                **{k: row[k] for k in ("scene_id", "role", "variant", "method", "query_id")},
                "region": region,
                "pixel_count": count,
                "total_valid": int(valid.sum()),
                "absrel_sum": total,
                "absrel_mean": total / count if count else None,
            }
        )
    return output, error


def observability_summary(scene_id, role, query_id, arrays, baseline_error):
    valid = arrays["query_valid"]
    counts = arrays["visible_view_count"][valid]
    errors = baseline_error[valid]
    rho = None
    if len(counts) >= 2 and np.ptp(counts) > 0 and np.ptp(errors) > 0:
        value = float(spearmanr(counts, errors).statistic)
        rho = value if np.isfinite(value) else None
    return {
        "scene_id": scene_id,
        "role": role,
        "query_id": query_id,
        "total_valid": int(valid.sum()),
        "region_counts": {k: int(v.sum()) for k, v in region_masks(arrays).items()},
        "spearman_visible_count_absrel": rho,
        "spearman_method": "S0 only; all GT-valid pixels; null for constant or undefined",
        "empty_masks_preserved": True,
    }


def verify_previous(path, seal):
    key = str(path)
    item = seal.get(key)
    if item is None:
        resolved = path.resolve()
        item = next((v for k, v in seal.items() if Path(k).resolve() == resolved), None)
    if item is None or sha(path) != item["sha256"]:
        raise PermissionError(f"Previous artifact missing from seal or changed: {path}")
    return item["sha256"]


def sealed_formal_barrier(root, manifest, plan):
    checked_manifest, checked_plan, config, specs, paths = ensure_phase(root, "formal")
    if (
        checked_manifest != manifest
        or checked_plan != plan
        or len(specs) != 136
        or len(manifest["scenes"]) != 17
    ):
        raise PermissionError("Exactly all136 new states / seventeen scenes must be sealed")
    barrier = StateSealBarrier(s["key"] for s in specs)
    hashes = {str(p): sha(p) for p in paths}
    hashes.update(config["source_sha256"])
    for spec in specs:
        state = torch.load(
            resolve(root, spec["state_path"]), map_location="cpu", weights_only=False
        )
        summary = read(resolve(root, spec["optimization_path"]) / "summary.json")
        if (
            hash_scene_state(state) != summary["state_hash"]
            or state.density_logits.shape[-3:] != (16, 16, 16)
            or not torch.equal(
                state.bounds.cpu(),
                torch.tensor(spec["bounds"], dtype=state.bounds.dtype).reshape(1, 2, 3),
            )
        ):
            raise PermissionError("Formal tensor state or bounds differs from sealed plan")
        barrier.seal(spec["key"], spec["scene_id"], state)
    barrier.assert_ready()
    return barrier, hashes


def load_references(root, manifest, access):
    seal = read(root / "audit/previous_experiment_seal.json")["files"]
    rows = []
    for name, variant in (
        ("context_results.json", "OLD_BASELINE"),
        ("oracle_results.json", "ORACLE"),
    ):
        old_raw_path = PREVIOUS / "raw" / name
        verify_previous(old_raw_path, seal)
        for old in read(old_raw_path):
            if not (
                old["bounds_mode"] == "FROZEN_GT_FREE_BOUNDS"
                and old["budget"] == 10000
                and old["selection"] == "FIXED_BUDGET"
                and old["method"] == "direct"
            ):
                continue
            path = resolve(PREVIOUS, old["prediction_path"])
            rows.append((old, path, variant))
    baseline_path = CAPACITY / "baseline_results.json"
    verify_previous(baseline_path, seal)
    for old in read(baseline_path):
        if old["method"] not in ("A", "B", "anchor"):
            continue
        roles = (old["method"],) if old["method"] in ("A", "B") else ("A", "B")
        for role in roles:
            rows.append(
                (
                    {**old, "role": role},
                    CAPACITY
                    / "raw/baseline_predictions"
                    / f"{old['scene_id']}_{old['query_id']}_{old['method']}.npz",
                    "ANCHOR" if old["method"] == "anchor" else "CARRIER",
                )
            )
    allowed = {r["scene_id"] for r in manifest["scenes"]}
    dest = root / "raw/reference_predictions"
    dest.mkdir()
    output = []
    for old, path, variant in rows:
        if old["scene_id"] not in allowed:
            raise PermissionError("Reference outside exposed roster")
        digest = verify_previous(path, seal)
        target = dest / f"{old['scene_id']}_{old['role']}_{old['query_id']}_{variant}.npz"
        shutil.copyfile(path, target)
        if sha(target) != digest:
            raise RuntimeError("Prediction copy hash mismatch")
        row = {
            **old,
            "variant": variant,
            "method": "direct",
            "key": f"{old['scene_id']}__{old['role']}__{variant}",
            "prediction_path": str(target.relative_to(root)),
            "prediction_file_sha256": digest,
            "reference_source": str(path),
            "reference_metrics": "inherited exact sealed matching row",
        }
        output.append(row)
        access.append(
            {
                "scene_id": old["scene_id"],
                "query_id": old["query_id"],
                "purpose": "POST_SEAL_REFERENCE_PREDICTION_COPY",
                "path": str(path),
                "sha256": digest,
            }
        )
    return output


def run(root):
    root = Path(root)
    manifest, plan = read(root / "scene_manifest.json"), read(root / "state_plan.json")
    destination = root / "raw/observability"
    if destination.exists() or (root / "raw/region_results.json").exists():
        raise FileExistsError("Do not overwrite prior observability diagnostics")
    barrier, input_hashes = sealed_formal_barrier(root, manifest, plan)
    query_path = root / "raw/query_results.json"
    query_rows = read(query_path)
    metadata_paths = [
        query_path,
        root / "raw/formal_evaluation_integrity.json",
        root / "observability_contract.json",
        root / "audit/previous_experiment_seal.json",
    ]
    input_hashes.update({str(p): sha(p) for p in metadata_paths})
    actual = [
        (r["scene_id"], r["role"], r["variant"], r["query_id"])
        for r in query_rows
        if r["method"] == "direct"
    ]
    expected = {
        (r["scene_id"], role, variant, fid)
        for r in manifest["scenes"]
        for role in ("A", "B")
        for variant in ("S0", "S1", "S2", "S3")
        for fid in r["roles"]["primary_query"]
    }
    if set(actual) != expected or len(actual) != len(expected):
        raise PermissionError("Complete exact formal direct-query roster required")
    if read(root / "raw/formal_evaluation_integrity.json").get("status") != "PASS":
        raise PermissionError("Formal post-seal query evaluation must be complete")
    access = []
    references = load_references(root, manifest, access)
    rows = [r for r in query_rows if r["method"] == "direct"] + references
    destination.mkdir()
    regions, summaries = [], []
    allowed = [r["scene_id"] for r in manifest["scenes"]]
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        kwargs = dict(
            capacity_scene_ids=allowed,
            holdout_scene_ids=manifest["FINAL_HOLDOUT_PROHIBITED"],
            access_log=access,
        )
        evaluator = CapacityEvaluator(
            record, manifest["image_size"], root, barrier=barrier, **kwargs
        )
        loader = ContextOnlyLoader(record, manifest["image_size"], root, track="RGBD", **kwargs)
        for role in ("A", "B"):
            first = len(access)
            context = loader.context(role)
            spec = next(
                s
                for s in plan["states"]
                if s["scene_id"] == sid and s["role"] == role and s["variant"] == "S0"
            )
            for fid in record["roles"]["primary_query"]:
                query = evaluator.query(fid)
                arrays = audit_context_observability(
                    barrier=barrier,
                    scene_id=sid,
                    query=query,
                    context=context,
                    bounds=torch.tensor(spec["bounds"], dtype=query.cameras.dtype),
                    anchor_c2w=context.cameras.c2w[0, 0],
                    allowed_scene_ids=allowed,
                    holdout_scene_ids=manifest["FINAL_HOLDOUT_PROHIBITED"],
                )
                path = destination / f"{sid}_{role}_{fid}.npz"
                np.savez_compressed(path, **arrays)
                matched = [
                    r
                    for r in rows
                    if r["scene_id"] == sid and r["role"] == role and r["query_id"] == fid
                ]
                if len(matched) != 8 or {r["variant"] for r in matched} != {
                    "S0",
                    "S1",
                    "S2",
                    "S3",
                    "ORACLE",
                    "OLD_BASELINE",
                    "CARRIER",
                    "ANCHOR",
                }:
                    raise PermissionError("Every locked query requires all variants and references")
                for row in matched:
                    predpath = resolve(root, row["prediction_path"])
                    digest = sha(predpath)
                    if row.get("prediction_file_sha256") != digest:
                        raise PermissionError("Saved prediction differs from sealed evaluation row")
                    input_hashes[str(predpath)] = digest
                    with np.load(predpath) as prediction:
                        values, errors = region_rows(row, prediction["depth"], arrays)
                    regions.extend(values)
                    if row["variant"] == "S0":
                        summaries.append(observability_summary(sid, role, fid, arrays, errors))
            for entry in access[first:]:
                entry["purpose"] = "DIAGNOSTIC_REGION_LABELING"
    barrier.assert_ready()
    if any(sha(Path(p)) != h for p, h in input_hashes.items()):
        raise PermissionError("Frozen state or prediction changed during observability audit")
    write_json(root / "raw/reference_results.json", references)
    write_json(root / "raw/region_results.json", regions)
    write_json(root / "raw/observability_summary.json", summaries)
    write_json(root / "raw/observability_GT_access.json", access)
    write_json(root / "raw/observability_seal_events.json", barrier.events)
    write_json(
        root / "audit/observability_integrity.json",
        {
            "status": "PASS",
            "all136_new_states_sealed_before_diagnostic": True,
            "input_sha256": input_hashes,
            "n_region_rows": len(regions),
            "n_role_query_labels": len(summaries),
            "n_reference_rows": len(references),
            "diagnostic_only": True,
            "model_state_unchanged": True,
            "FINAL_HOLDOUT_TOUCHED": False,
        },
    )
    return len(regions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.root))
