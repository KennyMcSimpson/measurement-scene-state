"""V11 Amendment 1 DEV evaluation: the frozen V11 evaluator plus float-rounding RGB clip.

Identical to the frozen V3 evaluator except that states are built from RGB-D
contexts: context depth arrives through the context-only RGBD loader, which holds
only context frames; query depth is still read only after every state is sealed."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import clone_scene_state, hash_scene_state, hash_value
from mcss.geometry import generate_rays, intersect_aabb, transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import (
    CapacityEvaluator,
    ContextOnlyLoader,
    StateSealBarrier,
)
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.geometry_carrier_experiment import validate_split
from mcss.mechanism_pilot.rgbd_completion_carrier import build_state, scene_data
from mcss.mechanism_pilot.small_training import sha, write_json

RGB_ROUNDING_TOLERANCE = 1e-6  # V11 Amendment 1: about 8 float32 ULP at 1.0


def clip_rounding(pred):
    """Clip rendered RGB into [0, 1] when it leaves the interval only by float32 rounding.

    Values inside [0, 1] are returned unchanged (bit-identical); an excursion larger than
    RGB_ROUNDING_TOLERANCE raises; non-finite values are left for the frozen metric guard.
    """
    rgb = pred["rgb"]
    finite = rgb[torch.isfinite(rgb)]
    excursion = (
        max(float((finite - 1).max()), float((-finite).max()), 0.0) if finite.numel() else 0.0
    )
    if excursion > RGB_ROUNDING_TOLERANCE:
        raise ValueError(f"Rendered RGB leaves [0, 1] by {excursion}, beyond float rounding")
    if excursion == 0.0:
        return pred
    return {**pred, "rgb": torch.where(torch.isfinite(rgb), rgb.clamp(0.0, 1.0), rgb)}


def _control(state, method):
    result = clone_scene_state(state)
    if method == "zero":
        result.density_logits.fill_(-1e6)
        result.color.zero_()
        if result.features is not None:
            result.features.zero_()
    elif method == "spatial_shuffle":
        voxels = math.prod(result.color.shape[2:])  # 16**3 reproduces V7 exactly
        permutation = torch.randperm(voxels, generator=torch.Generator().manual_seed(20260927))
        permutation = permutation.to(result.color.device)
        for name in ("density_logits", "color", "log_variance", "features"):
            value = getattr(result, name)
            if value is not None:
                setattr(result, name, value.flatten(2)[:, :, permutation].reshape_as(value))
    else:
        raise ValueError(method)
    return result


def _extra(state, camera, pred, rgb):
    origins, directions = generate_rays(camera)
    _, _, hit = intersect_aabb(origins, directions, state.bounds[0])
    opacity = pred["visibility"][:, :, 0]
    n = int(hit.sum())
    return {
        "ray_hitfraction": float(hit.float().mean()),
        "hit_count": n,
        "hit_opacity": float(opacity[hit].mean()) if n else None,
        "gray_rgb_mse": float((rgb - 0.5).square().mean()),
    }


def selection_summary(query_rows, context_rows, step):
    direct = [r for r in query_rows if r["method"] == "direct"]
    contexts = [r for r in context_rows if r["method"] == "direct"]
    scene_ids = sorted({r["scene_id"] for r in direct})
    if not scene_ids:
        raise ValueError("Nonempty DEV direct predictions required")

    def equal_mean(rows, field):
        return float(
            np.mean([np.mean([r[field] for r in rows if r["role"] == role]) for role in ("A", "B")])
        )

    scenes = []
    for sid in scene_ids:
        rows = [r for r in direct if r["scene_id"] == sid]
        cr = [r for r in contexts if r["scene_id"] == sid]
        hits = sum(r["hit_count"] for r in rows)
        hit_opacity = (
            sum(r["hit_opacity"] * r["hit_count"] for r in rows if r["hit_count"]) / hits
            if hits
            else None
        )
        coverage, geometric = equal_mean(rows, "coverage"), equal_mean(rows, "ray_hitfraction")
        values = {
            "scene_id": sid,
            "query_absrel": equal_mean(rows, "depth_absrel"),
            "context_absrel": equal_mean(cr, "depth_absrel"),
            "rgb_mse": equal_mean(rows, "rgb_mse"),
            "gray_rgb_mse": equal_mean(rows, "gray_rgb_mse"),
            "coverage": coverage,
            "ray_hitfraction": geometric,
            "hit_opacity": hit_opacity,
            "hit_count": hits,
            "no_hit_rays": hits == 0,
            "coverage_eligible": coverage + 1e-12 >= 0.99 * geometric,
            "hit_opacity_eligible": hit_opacity is None or hit_opacity >= 0.05,
        }
        values["generalization_gap"] = values["query_absrel"] - values["context_absrel"]
        scenes.append(values)
    rgb = float(np.mean([r["rgb_mse"] for r in scenes]))
    gray = float(np.mean([r["gray_rgb_mse"] for r in scenes]))
    finite = all(
        np.isfinite(v)
        for row in direct + contexts
        for v in row.values()
        if isinstance(v, (float, int))
    )
    return {
        "step": step,
        "eligible": bool(
            finite
            and rgb <= 1.25 * gray
            and all(r["coverage_eligible"] and r["hit_opacity_eligible"] for r in scenes)
        ),
        "finite": bool(finite),
        "rgb_eligible": rgb <= 1.25 * gray,
        "query_absrel": float(np.mean([r["query_absrel"] for r in scenes])),
        "context_absrel": float(np.mean([r["context_absrel"] for r in scenes])),
        "generalization_gap": float(np.mean([r["generalization_gap"] for r in scenes])),
        "rgb_mse": rgb,
        "gray_rgb_mse": gray,
        "scenes": scenes,
        "no_hit_scenes": [r["scene_id"] for r in scenes if r["no_hit_rays"]],
        "coverage_rule": "scene mean coverage >= .99 * geometric ray_hitfraction",
        "opacity_rule": "hit-count-weighted mean >= .05; no-hit scene recorded, not collapse",
        "selection_unit": "scene equal; within scene roles then queries equal",
    }


@torch.no_grad()
def evaluate_model(carrier, manifest, root, out, variant, seed, step, device, diagnostics=False):
    """Ordinary DEV entry only; fresh qualification requires a separate locked runner."""
    validate_split(manifest)
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "DEV"}
    if not records:
        raise PermissionError("DEV scenes required; this entry never evaluates FRESH_QUALIFICATION")
    if variant not in ("C0", "C1", "C2"):
        raise ValueError("Unknown matched carrier variant")
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError("Refusing to overwrite prior checkpoint evaluation")
    forbidden = set(manifest.get("protected_scene_ids", [])) | {
        r["scene_id"] for r in manifest["scenes"] if r["split"] != "DEV"
    }
    keys = [f"{sid}/{name}" for sid in records for name in ("A", "B", "anchor_A", "anchor_B")]
    barrier = StateSealBarrier(keys)
    access, metadata = [], {}
    before = hash_value(carrier.state_dict())
    was_training = carrier.training
    carrier.eval()
    for sid, record in records.items():
        data = scene_data(variant)(record, manifest, root, device, access)
        for name in ("A", "B", "anchor_A", "anchor_B"):
            role = name[-1]
            context = data.context("anchor" if name.startswith("anchor") else role)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            construction_started = time.perf_counter()
            state, anchor = build_state(carrier, context, data.bounds[role], f"dev:{sid}:{name}")
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            construction_seconds = time.perf_counter() - construction_started
            if any(
                not torch.isfinite(getattr(state, k)).all()
                for k in ("density_logits", "color", "log_variance", "features")
            ):
                raise FloatingPointError("Nonfinite carrier state; cannot qualify checkpoint")
            key = f"{sid}/{name}"
            barrier.seal(key, sid, state.to(device="cpu"))
            metadata[key] = {
                "scene_id": sid,
                "role": role,
                "construction": name,
                "anchor_c2w": anchor.detach().cpu().tolist(),
                "state_hash": hash_scene_state(state),
                "bounds": state.bounds.cpu().tolist(),
                "context_frame_ids": [o.frame_id for o in context],
                "construction_seconds": construction_seconds,
            }
    barrier.assert_ready()
    assert hash_value(carrier.state_dict()) == before
    # RGB-D track: construction reads depth only through the context-only loader.
    assert all(
        e.get("purpose") == "CONTEXT_ONLY_RGBD" for e in access if "depth" in e.get("channels", [])
    )
    seal_boundary = len(access)
    access.append({"event": "ALL_DEV_STATES_SEALED", "state_count": len(keys)})
    out.mkdir(parents=True)
    torch.save({key: barrier.state(key) for key in keys}, out / "states.pt")
    write_json(out / "state_hashes.json", metadata)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    query_rows, context_rows = [], []
    if diagnostics:
        (out / "predictions").mkdir()
    evaluators = {
        sid: CapacityEvaluator(
            record,
            manifest["image_size"],
            root,
            device,
            barrier=barrier,
            capacity_scene_ids=tuple(records),
            holdout_scene_ids=tuple(forbidden),
            access_log=access,
        )
        for sid, record in records.items()
    }
    ids = sorted(records)
    started = time.perf_counter()
    for key in keys:
        meta = metadata[key]
        sid, role, name = meta["scene_id"], meta["role"], meta["construction"]
        direct = not name.startswith("anchor")
        state = barrier.state(key).to(device=device)
        inverse = torch.linalg.inv(
            torch.tensor(meta["anchor_c2w"], dtype=torch.float32, device=device)
        )
        common = {
            "variant": variant,
            "seed": seed,
            "step": step,
            "scene_id": sid,
            "role": role,
            "construction": name,
            "state_key": key,
        }
        # Context labels are evaluator-only, read after *all* DEV states are sealed.
        if direct:
            start = len(access)
            batch = ContextOnlyLoader(
                records[sid],
                manifest["image_size"],
                root,
                device,
                track="RGBD",
                capacity_scene_ids=tuple(records),
                holdout_scene_ids=tuple(forbidden),
                access_log=access,
            ).context(role)
            for event in access[start:]:
                event["purpose"] = "POST_ALL_DEV_SEAL_CONTEXT_EVALUATION"
            cameras = transform_cameras(batch.cameras, inverse)
            for j, fid in enumerate(batch.frame_ids):
                camera = cameras.select_views(torch.tensor([j], device=device))
                pred = clip_rounding(renderer(state, camera, ("rgb", "depth", "visibility")))
                rgb, depth = batch.rgb[:, j : j + 1], batch.depth[:, j : j + 1]
                context_rows.append(
                    {
                        **common,
                        "method": "direct",
                        "frame_id": fid,
                        "used_state_hash": hash_scene_state(state),
                        "camera_hash": hash_value(camera),
                        **_metrics(pred, rgb, depth),
                        **_extra(state, camera, pred, rgb),
                    }
                )
        for fid in records[sid]["roles"]["primary_query"]:
            batch = evaluators[sid].query(fid)
            camera = transform_cameras(batch.cameras, inverse)
            camera_hash = hash_value(camera)
            methods = ["direct" if direct else "anchor"]
            if diagnostics and direct:
                methods += ["wrong_scene", "spatial_shuffle", "zero"]
            for method in methods:
                donor = None
                used = state
                if method == "wrong_scene":
                    if len(ids) < 2:
                        raise PermissionError("At least two DEV scenes required")
                    donor = ids[(ids.index(sid) + 1) % len(ids)]
                    used = barrier.state(f"{donor}/{role}").to(device=device)
                elif method in ("spatial_shuffle", "zero"):
                    used = _control(state, method)
                state_hash = hash_scene_state(used)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                pred = clip_rounding(renderer(used, camera, ("rgb", "depth", "visibility")))
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                row = {
                    **common,
                    "method": method,
                    "query_id": fid,
                    "donor_scene_id": donor,
                    "used_state_hash": state_hash,
                    "query_camera_hash": camera_hash,
                    "seconds": elapsed,
                    "prediction_hash": hash_value(pred),
                    **_metrics(pred, batch.rgb, batch.depth),
                    **_extra(used, camera, pred, batch.rgb),
                }
                if diagnostics:
                    path = out / "predictions" / f"{sid}_{name}_{fid}_{method}.npz"
                    np.savez_compressed(
                        path,
                        rgb=pred["rgb"][0, 0].cpu().numpy(),
                        depth=pred["depth"][0, 0, 0].cpu().numpy(),
                        opacity=pred["visibility"][0, 0, 0].cpu().numpy(),
                    )
                    row["prediction_path"] = str(path.relative_to(out))
                    row["prediction_file_sha256"] = sha(path)
                query_rows.append(row)
                assert hash_scene_state(used) == state_hash and hash_value(camera) == camera_hash
            barrier.assert_ready()
    selection = selection_summary(query_rows, context_rows, step)
    for name, value in [
        ("query_results", query_rows),
        ("context_results", context_rows),
        ("selection", selection),
        ("GT_access", access),
        ("seal_events", barrier.events),
    ]:
        write_json(out / f"{name}.json", value)
    barrier.assert_ready()
    assert hash_value(carrier.state_dict()) == before
    carrier.train(was_training)
    write_json(
        out / "integrity.json",
        {
            "status": "PASS",
            "variant": variant,
            "seed": seed,
            "step": step,
            "states": len(keys),
            "query_rows": len(query_rows),
            "context_rows": len(context_rows),
            "carrier_state_dict_hash": before,
            "states_file_sha256": sha(out / "states.pt"),
            "all_states_sealed_before_GT": True,
            "preseal_RGB_camera_access_count": seal_boundary,
            "test_time_depth_used": True,
            "fresh_qualification_opened": False,
            "diagnostics": diagnostics,
            "anchor_definition": (
                "anchor RGB + depth + camera; full-role camera-derived bounds matched"
            ),
            "seconds": time.perf_counter() - started,
        },
    )
    return {"query_rows": query_rows, "context_rows": context_rows, "selection": selection}
