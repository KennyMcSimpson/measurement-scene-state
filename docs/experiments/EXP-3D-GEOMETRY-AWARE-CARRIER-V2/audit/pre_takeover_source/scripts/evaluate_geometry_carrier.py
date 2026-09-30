#!/usr/bin/env python3
"""Evaluate frozen per-seed DEV selections and derive post-seal diagnostics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from mcss.mechanism_pilot.context_observability import audit_context_observability
from mcss.mechanism_pilot.direct_capacity_contracts import (
    CapacityEvaluator,
    ContextOnlyLoader,
    StateSealBarrier,
)
from mcss.mechanism_pilot.geometry_carrier_evaluation import evaluate_model
from mcss.mechanism_pilot.geometry_carrier_experiment import load_checkpoint, read, verify_lock
from mcss.mechanism_pilot.small_training import sha, write_json


def evaluate(root, device):
    root = Path(root).resolve()
    manifest, config = verify_lock(root)
    torch.set_num_threads(config["num_threads"])
    selected = {variant: {} for variant in config["variants"]}
    summaries = []
    for variant in config["variants"]:
        for seed in config["seeds"]:
            run = root / "checkpoints" / f"{variant}_{seed}"
            summary = read(run / "summary.json")
            summaries.append(summary)
            item = summary["selection"]
            if item["status"] != "SELECTED":
                write_json(
                    root / "selection_blocked.json",
                    {
                        "status": "NO_ELIGIBLE_CHECKPOINT",
                        "variant": variant,
                        "seed": seed,
                        "fresh_opened": False,
                    },
                )
                raise PermissionError("No eligible checkpoint; refusing fallback")
            if sha(Path(item["checkpoint"])) != item["checkpoint_sha256"]:
                raise PermissionError("Selected checkpoint changed")
            selected[variant][str(seed)] = {
                "path": item["checkpoint"],
                "step": item["selected_step"],
                "sha256": item["checkpoint_sha256"],
            }
    for seed in config["seeds"]:
        pair = [s for s in summaries if s["seed"] == seed]
        if (
            len({s["initial_state_dict_hash"] for s in pair}) != 1
            or len({s["data_ray_stream_sha256"] for s in pair}) != 1
        ):
            raise PermissionError("Matched initialization/sampling violated")
    write_json(root / "selected_checkpoints.json", selected)
    write_json(
        root / "selection_freeze.json",
        {
            "selected_sha256": sha(root / "selected_checkpoints.json"),
            "primary_method": "C1",
            "all_seeds_retained": True,
            "fresh_opened": False,
        },
    )
    rows = []
    contexts = []
    training = []
    devcurves = []
    cost = {}
    for variant in config["variants"]:
        costs = []
        inference_times = []
        for seed in config["seeds"]:
            run = root / "checkpoints" / f"{variant}_{seed}"
            item = selected[variant][str(seed)]
            model, payload = load_checkpoint(item["path"], device)
            if payload["lock_sha256"] != sha(root / "preregistration.json"):
                raise PermissionError("Checkpoint not from frozen training")
            out = root / "raw" / "selected_dev" / f"{variant}_{seed}"
            result = evaluate_model(
                model,
                manifest,
                root,
                out,
                variant=variant,
                seed=seed,
                step=item["step"],
                device=device,
                diagnostics=True,
            )
            inference_times.extend(
                m["construction_seconds"]
                for m in read(out / "state_hashes.json").values()
                if m["construction"] in ("A", "B")
            )
            for row in result["query_rows"]:
                row["prediction_path"] = str((out / row["prediction_path"]).relative_to(root))
                rows.append(row)
            contexts.extend(result["context_rows"])
            training.extend(
                {"variant": variant, "seed": seed, **__import__("json").loads(line)}
                for line in (run / "training.jsonl").read_text().splitlines()
            )
            devcurves.extend(
                {"variant": variant, "seed": seed, **r, "depth_absrel": r["query_absrel"]}
                for r in read(run / "dev_curve.json")
            )
            costs.append(read(run / "summary.json"))
        cost[variant] = {
            "training_seconds": sum(s["wall_seconds"] for s in costs),
            "inference_seconds_per_state": float(np.mean(inference_times)),
            "peak_cuda_allocated_bytes": max((s["peak_cuda_allocated_bytes"] or 0) for s in costs),
            "checkpoint_bytes": sum(
                Path(v["path"]).stat().st_size for v in selected[variant].values()
            ),
            "renderer_seconds_per_query": float(
                np.mean(
                    [
                        r["seconds"]
                        for r in rows
                        if r["variant"] == variant and r["method"] == "direct"
                    ]
                )
            ),
            "seconds_per_training_step": float(
                np.mean([r["seconds"] for r in training if r["variant"] == variant])
            ),
        }
    write_json(root / "raw/dev_query_results.json", rows)
    write_json(root / "raw/dev_context_results.json", contexts)
    write_json(root / "training_curves.json", {"training": training, "dev": devcurves})
    write_json(
        root / "cost_analysis.json",
        {
            "variants": cost,
            "gpu_shared": True,
            "inference_architecture_identical": True,
            "cost_interpretation": "training wall includes DEV "
            "selection and checkpoint IO; step seconds separate; surface is training-only",
        },
    )
    for variant, name in [("C0", "baseline_training"), ("C1", "surface_training")]:
        write_json(root / f"{name}.json", [s for s in summaries if s["variant"] == variant])
    # All selected prediction sets exist before diagnostics; reconstruct a CPU seal barrier.
    states = {}
    for variant in config["variants"]:
        for seed in config["seeds"]:
            values = torch.load(
                root / "raw/selected_dev" / f"{variant}_{seed}" / "states.pt",
                map_location="cpu",
                weights_only=False,
            )
            for key, state in values.items():
                states[f"{variant}:{seed}:{key}"] = state
    barrier = StateSealBarrier(states)
    for key, state in states.items():
        sid = key.split(":", 2)[2].split("/")[0]
        barrier.seal(key, sid, state)
    barrier.assert_ready()
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "DEV"}
    forbidden = set(manifest["protected_scene_ids"]) | {
        r["scene_id"] for r in manifest["scenes"] if r["split"] != "DEV"
    }
    access = []
    arrays = {}
    maskdir = root / "raw/observability"
    maskdir.mkdir()
    for sid, record in records.items():
        evaluator = CapacityEvaluator(
            record,
            manifest["image_size"],
            root,
            "cpu",
            barrier=barrier,
            capacity_scene_ids=tuple(records),
            holdout_scene_ids=tuple(forbidden),
            access_log=access,
        )
        loader = ContextOnlyLoader(
            record,
            manifest["image_size"],
            root,
            "cpu",
            track="RGBD",
            capacity_scene_ids=tuple(records),
            holdout_scene_ids=tuple(forbidden),
            access_log=access,
        )
        anchor = torch.tensor(
            next(f for f in record["frames"] if f["frame_id"] == record["roles"]["context_a"][0])[
                "c2w"
            ],
            dtype=torch.float32,
        )
        for role in ("A", "B"):
            context = loader.context(role)
            state = states[f"C0:{config['seeds'][0]}:{sid}/{role}"]
            for qid in record["roles"]["primary_query"]:
                query = evaluator.query(qid)
                arr = audit_context_observability(
                    barrier=barrier,
                    scene_id=sid,
                    query=query,
                    context=context,
                    bounds=state.bounds,
                    anchor_c2w=anchor,
                    allowed_scene_ids=tuple(records),
                    holdout_scene_ids=tuple(forbidden),
                )
                arrays[sid, role, qid] = arr
                np.savez_compressed(maskdir / f"{sid}_{role}_{qid}.npz", **arr)
    regions = []
    matched = []
    direct = {
        (r["variant"], r["seed"], r["scene_id"], r["role"], r["query_id"]): r
        for r in rows
        if r["method"] == "direct"
    }
    for (sid, role, qid), arr in arrays.items():
        valid = arr["query_valid"]
        gt = arr["query_gt_depth"].astype(np.float64)
        for seed in config["seeds"]:
            predictions = {
                v: dict(np.load(root / direct[v, seed, sid, role, qid]["prediction_path"]))
                for v in config["variants"]
            }
            common = valid.copy()
            for p in predictions.values():
                common &= p["opacity"] > 1e-6
            digest = hashlib.sha256(common.tobytes()).hexdigest()
            for v, p in predictions.items():
                identity = {
                    "variant": v,
                    "seed": seed,
                    "scene_id": sid,
                    "role": role,
                    "query_id": qid,
                    "method": "direct",
                }
                err = np.zeros(gt.shape)
                err[valid] = np.abs(p["depth"][valid] - gt[valid]) / gt[valid]
                norm = p["depth"].astype(np.float64) / np.maximum(p["opacity"], 1e-6)
                normalized = np.abs(norm[common] - gt[common]) / gt[common]
                matched.append(
                    {
                        **identity,
                        "depth_absrel": float(err[common].mean()) if common.any() else None,
                        "normalized_depth_absrel": float(normalized.mean())
                        if common.any()
                        else None,
                        "common_valid_count": int(common.sum()),
                        "total_gt_valid": int(valid.sum()),
                        "common_mask_hash": digest,
                    }
                )
                for i, name in enumerate(("OBS0", "OBS1", "OBS2PLUS")):
                    mask = valid & (arr["obs_class"] == i)
                    count = int(mask.sum())
                    total = float(err[mask].sum())
                    regions.append(
                        {
                            **identity,
                            "region": name,
                            "pixel_count": count,
                            "total_valid": int(valid.sum()),
                            "absrel_sum": total,
                            "absrel_mean": total / count if count else None,
                        }
                    )
    write_json(root / "raw/dev_matched_results.json", matched)
    write_json(root / "raw/dev_region_results.json", regions)
    write_json(root / "audit/diagnostic_GT_access.json", access)
    state_ok = all(
        len(
            {
                r["used_state_hash"]
                for r in rows
                if r["variant"] == v
                and r["seed"] == s
                and r["scene_id"] == sid
                and r["role"] == role
                and r["method"] == "direct"
            }
        )
        == 1
        for v in config["variants"]
        for s in config["seeds"]
        for sid in records
        for role in ("A", "B")
    )
    camera_ok = all(
        len(
            {
                r["query_camera_hash"]
                for r in rows
                if r["variant"] == v
                and r["seed"] == s
                and r["scene_id"] == sid
                and r["role"] == role
                and r["query_id"] == qid
            }
        )
        == 1
        for v in config["variants"]
        for s in config["seeds"]
        for sid, record in records.items()
        for role in ("A", "B")
        for qid in record["roles"]["primary_query"]
    )
    if not state_ok or not camera_ok:
        raise AssertionError("State/camera invariance failed")
    write_json(
        root / "audit/dev_state_use.json",
        {
            "status": "PASS",
            "shared_state_multiple_queries": state_ok,
            "query_after_state_seal": True,
            "recipient_camera_preserved": camera_ok,
            "test_time_depth_used": False,
            "fresh_qualification_opened": False,
            "matched_initialization_and_streams": True,
        },
    )
    verify_lock(root)
    print("SELECTED_DEV_EVALUATION_PASS", len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    evaluate(a.root, a.device)
