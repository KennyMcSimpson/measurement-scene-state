"""Static-only sealed carrier qualification and readout attribution; no write actions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.checkpoint import load_dynamic_checkpoint
from mcss.dynamic.feedback import anchored_observation, batched_camera
from mcss.dynamic.types import (
    OnlineObservation,
    SealedScene,
    clone_scene_state,
    hash_fast_content,
    hash_scene_state,
    hash_value,
)
from mcss.geometry import generate_rays, transform_cameras, transform_points
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.contracts import EvaluationLedger
from mcss.mechanism_pilot.small_training import AccessLog, TrainScene, sha, write_json
from mcss.mechanism_pilot.static_geometry import volume_geometry
from mcss.mechanism_pilot.static_residual import load_residual_checkpoint, zero_state_like
from mcss.mechanism_pilot.statistics import measurement_metrics, paired_scene_bootstrap
from mcss.mechanism_pilot.visibility import common_visibility_masks

METRICS = (
    "rgb_mse",
    "rgb_psnr",
    "rgb_ssim",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "opacity",
    "coverage",
    "valid_depth_fraction",
)


def validate_static_roles(scene, checkpoint_train_scenes, *, unseen=False):
    sid, roles = scene["scene_id"], scene["roles"]
    a, b = roles["context_a"], roles["context_b"]
    if not a or len(a) != len(b) or a[0] != b[0] or set(a) & set(b) != {a[0]}:
        raise ValueError("Contexts must share only the coordinate anchor and match sizes")
    primary = roles.get("primary_query", [8, 9])
    secondary = roles.get("secondary_query", [12, 13, 14, 15])
    if not primary or (set(a) | set(b)) & set(primary + secondary):
        raise PermissionError("Query cannot enter state construction")
    if set(primary) & set(secondary):
        raise ValueError("Query cohorts overlap")
    if unseen and (sid in checkpoint_train_scenes or scene.get("split") != "dev"):
        raise PermissionError("Unseen qualification overlaps checkpoint training or is not dev")
    if unseen and not scene.get("physical_scene_id"):
        raise PermissionError("Unseen scene requires audited source-asset identity")
    return {"primary": primary, "secondary": secondary}


@torch.no_grad()
def construct(carrier, observations, episode, *, shuffle=False):
    anchor = observations[0].camera.c2w
    cache = ObservationCache(episode, observations[0].scene_id)
    fast = carrier.initial_fast(episode)
    for observation in observations:
        arrived = anchored_observation(observation, anchor)
        feature = carrier.encode(arrived)
        if shuffle:
            generator = torch.Generator(device="cpu").manual_seed(20260927 + observation.frame_id)
            order = torch.randperm(feature.shape[-2] * feature.shape[-1], generator=generator).to(
                feature.device
            )
            feature = feature.flatten(1)[:, order].reshape_as(feature)
        cache = cache.append(arrived, feature)
    state = carrier.materialize(cache, fast)
    assert not fast.delta_fuse.any() and not fast.delta_complete.any()
    return state, anchor, fast


def summarize(rows):
    """Scene-equal summaries and paired deltas, cohorts kept separate."""
    result = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        subset = [r for r in rows if r["cohort"] == cohort]
        methods = {}
        for name in sorted({r["method"] for r in subset}):
            selected = [r for r in subset if r["method"] == name]
            scenes = sorted({r["scene_id"] for r in selected})
            methods[name] = {}
            for region in [
                "full_image",
                "common_visible",
                "inside_volume_supported",
                "inside_volume_unsupported",
                "outside_volume",
            ]:
                rr = (
                    selected
                    if region == "full_image"
                    else [
                        {"scene_id": r["scene_id"], **r[region]}
                        for r in selected
                        if r[region] is not None
                    ]
                )
                mm = {}
                for metric in METRICS:
                    per_scene = {}
                    for sid in scenes:
                        values = [r[metric] for r in rr if r["scene_id"] == sid]
                        per_scene[sid] = (
                            None
                            if not values or any(v is None for v in values)
                            else float(np.mean(values))
                        )
                    mm[metric] = {
                        "per_scene": per_scene,
                        "mean": None
                        if any(v is None for v in per_scene.values())
                        else float(np.mean(list(per_scene.values()))),
                    }
                methods[name][region] = {
                    "metrics": mm,
                    "eligible_queries": len(rr),
                    "total_queries": len(selected),
                }
        paired = {}
        base = {
            s: methods["A"]["full_image"]["metrics"]["depth_absrel"]["per_scene"][s] for s in scenes
        }

        def values(name, methods=methods):
            return methods[name]["full_image"]["metrics"]["depth_absrel"]["per_scene"]

        for label, left, right in [
            ("anchor_minus_A", "anchor", "A"),
            ("anchor_minus_B", "anchor", "B"),
            ("wrong_scene_damage", "wrong_scene", "A"),
            ("B_minus_A", "B", "A"),
        ]:
            left_values, right_values = values(left), values(right)
            paired[label] = paired_scene_bootstrap(
                {s: left_values[s] - right_values[s] for s in base}
            )
        paired["full_context_gain"] = paired_scene_bootstrap(
            {s: values("anchor")[s] - (values("A")[s] + values("B")[s]) / 2 for s in base}
        )
        if "Residual_A" in methods:
            for label, left, right in [
                ("residual_minus_fixed_A", "Residual_A", "A"),
                ("residual_minus_fixed_B", "Residual_B", "B"),
                ("residual_state_shuffle_damage", "Residual_shuffled", "Residual_A"),
                ("residual_zero_state_damage", "Residual_zero", "Residual_A"),
            ]:
                paired[label] = paired_scene_bootstrap(
                    {s: values(left)[s] - values(right)[s] for s in base}
                )
        result[cohort] = {
            "methods": methods,
            "paired_depth_absrel": paired,
            "common_region_note": (
                "Only nonempty query masks averaged; missing scenes produce null macro"
            ),
        }
    return result


@torch.no_grad()
def run_static(
    manifest_path, carrier_path, output_dir, *, residual_path=None, device="cuda", unseen=False
):
    manifest_path, carrier_path, output = map(Path, (manifest_path, carrier_path, output_dir))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite static evaluation")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    carrier, writer, metadata = load_dynamic_checkpoint(carrier_path, device=device)
    carrier.eval().requires_grad_(False)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    training_scenes = metadata["provenance"]["training_scenes"]
    scopes = {
        s["scene_id"]: validate_static_roles(s, training_scenes, unseen=unseen)
        for s in manifest["scenes"]
    }
    if unseen and len(manifest["scenes"]) < 6:
        raise PermissionError("At least six eligible unseen scenes required for qualification")
    if unseen and len({s["physical_scene_id"] for s in manifest["scenes"]}) != len(
        manifest["scenes"]
    ):
        raise PermissionError("Repeated physical/source asset in unseen cohort")
    residual = None
    if residual_path:
        residual, residual_meta = load_residual_checkpoint(
            residual_path, carrier_sha256=metadata["sha256"], device=device
        )
    input_hashes = {
        str(manifest_path.resolve()): sha(manifest_path),
        str(carrier_path.resolve()): sha(carrier_path),
    }
    if residual_path:
        input_hashes[str(Path(residual_path).resolve())] = sha(Path(residual_path))
    for record in manifest["scenes"]:
        for frame in record["frames"]:
            for key in ["rgb", "depth"]:
                path = Path(frame[key])
                path = path if path.is_absolute() else manifest_path.parent / path
                input_hashes[str(path.resolve())] = sha(path)
    source_hashes = {str(p): sha(p) for p in Path(__file__).resolve().parents[1].rglob("*.py")}
    write_json(
        output / "lock.json",
        {
            "checkpoint": metadata,
            "manifest_sha256": sha(manifest_path),
            "residual_sha256": sha(Path(residual_path)) if residual_path else None,
            "carrier": asdict(carrier.config),
            "scopes": scopes,
            "unseen": unseen,
            "input_sha256": input_hashes,
            "source_sha256": source_hashes,
            "dynamic_ttt_run": False,
        },
    )
    access = AccessLog(output / "label_access.jsonl")
    scenes = {
        r["scene_id"]: TrainScene(r, manifest["image_size"], manifest_path.parent, device, [])
        for r in manifest["scenes"]
    }
    states, observations, state_diffs = {}, {}, []
    start = time.perf_counter()
    config_hash = hash_value({"carrier": asdict(carrier.config), "samples": 64})
    for sid, scene in scenes.items():
        roles = scene.record["roles"]
        obs = {}
        for fid in sorted(set(roles["context_a"] + roles["context_b"])):
            f = scene.frames[fid]
            access.append(
                {
                    "scene_id": sid,
                    "frame_id": fid,
                    "kind": "arrived_rgb_camera",
                    "stage": "construction",
                }
            )
            obs[fid] = OnlineObservation(sid, fid, scene.rgb(f), scene.camera(f))
        observations[sid] = obs
        plans = [
            ("A", roles["context_a"], False),
            ("B", roles["context_b"], False),
            ("anchor", roles["context_a"][:1], False),
            ("feature_shuffle", roles["context_a"], True),
        ]
        materialized = {}
        for name, ids, shuffle in plans:
            state, anchor, fast = construct(
                carrier, [obs[i] for i in ids], f"{sid}-{name}", shuffle=shuffle
            )
            materialized[name] = state
            states[f"{sid}/{name}"] = SealedScene(
                f"{sid}-{name}",
                sid,
                scene.record["split"],
                "static-query",
                state,
                hash_scene_state(state),
                tuple(ids),
                metadata["sha256"],
                config_hash,
                hash_fast_content(fast),
                anchor.clone(),
            )
        for name in ["channel_permutation", "state_mask"]:
            state = (
                clone_scene_state(materialized["A"])
                if name == "channel_permutation"
                else zero_state_like(materialized["A"])
            )
            if name == "channel_permutation":
                state.features.copy_(state.features.flip(1))
            states[f"{sid}/{name}"] = SealedScene(
                f"{sid}-{name}",
                sid,
                scene.record["split"],
                "static-query",
                state,
                hash_scene_state(state),
                tuple(roles["context_a"]),
                metadata["sha256"],
                config_hash,
                hash_fast_content(fast),
                anchor.clone(),
            )
        state_diffs.append(
            {
                "scene_id": sid,
                "A_equals_anchor": hash_scene_state(materialized["A"])
                == hash_scene_state(materialized["anchor"]),
                "diagnostic_only": ["feature_shuffle", "channel_permutation", "state_mask"],
                "channel_permutation_note": (
                    "Feature-only negative control: fixed renderer reads typed fields, "
                    "residual can read features"
                ),
            }
        )
    ledger = EvaluationLedger(states.keys())
    for key, state in states.items():
        ledger.seal(key, state)
    write_json(
        output / "state_manifest.json",
        {
            k: {"state_hash": s.state_hash, "observed_ids": s.observed_ids}
            for k, s in states.items()
        },
    )
    torch.save(states, output / "sealed_states.pt")
    write_json(output / "state_diagnostics.json", state_diffs)
    rows = []
    ids = list(scenes)
    candidate = carrier._candidate_points
    bounds = carrier._bounds.reshape(2, 3)
    radius = float(((bounds[1] - bounds[0]) / torch.tensor([8, 8, 8], device=device)).norm() / 2)
    for ix, (sid, scene) in enumerate(scenes.items()):
        obs, roles = observations[sid], scene.record["roles"]

        def depth_load(fid, kind, sid=sid, scene=scene):
            access.append(
                {"scene_id": sid, "frame_id": fid, "kind": kind, "stage": "sealed_evaluator"}
            )
            return (
                torch.from_numpy(np.load(scene.path(scene.frames[fid], "depth")))
                .float()
                .to(device)
                .squeeze()
            )

        contexts = {}
        for role in ["context_a", "context_b"]:
            contexts[role] = [
                (
                    obs[i].camera,
                    ledger.read_query_ground_truth(
                        f"visibility-{sid}-{i}",
                        lambda i=i, depth_load=depth_load: depth_load(
                            i, "context_GT_visibility_only"
                        ),
                    ),
                )
                for i in roles[role]
            ]
        anchor = states[f"{sid}/A"].anchor_c2w
        from mcss.geometry import project_world

        support = (
            torch.stack(
                [
                    project_world(candidate, anchored_observation(obs[i], anchor).camera)[2]
                    for i in roles["context_a"]
                ]
            ).sum(0)
            >= 2
        )
        for cohort, query_ids in scopes[sid].items():
            for fid in query_ids:
                frame = scene.frames[fid]

                def camera_loader(sid=sid, fid=fid, scene=scene, frame=frame):
                    access.append(
                        {
                            "scene_id": sid,
                            "frame_id": fid,
                            "kind": "query_camera",
                            "stage": "sealed_evaluator",
                        }
                    )
                    return scene.camera(frame)

                world_camera = ledger.read_query_camera(f"{sid}-{fid}", camera_loader)

                def truth_loader(fid=fid, scene=scene, frame=frame, depth_load=depth_load):
                    depth = depth_load(fid, "query_GT_rgb_depth")
                    return scene.rgb(frame), depth

                truth, depth = ledger.read_query_ground_truth(f"{sid}-{fid}", truth_loader)
                common = common_visibility_masks(
                    world_camera, depth, contexts["context_a"], contexts["context_b"]
                )["depth_consistent_common"]
                camera = batched_camera(transform_cameras(world_camera, torch.linalg.inv(anchor)))
                camera_hash = hash_value(camera)
                origins, directions = generate_rays(world_camera)
                valid = torch.isfinite(depth) & (depth > 0)
                points = transform_points(
                    (origins + directions * torch.where(valid, depth, 0)[..., None]).reshape(-1, 3),
                    torch.linalg.inv(anchor),
                )
                outside, nearest, nearest_supported, has_support = volume_geometry(
                    points.cpu(), candidate.cpu(), bounds.cpu(), support.cpu(), radius
                )
                inside = (outside == 0).reshape(depth.shape).to(device) & valid
                supported = has_support.reshape(depth.shape).to(device) & valid
                regions = {
                    "common_visible": common,
                    "inside_volume_supported": inside & supported,
                    "inside_volume_unsupported": inside & ~supported,
                    "outside_volume": valid & ~inside,
                }
                methods = [
                    ("A", f"{sid}/A", False),
                    ("B", f"{sid}/B", False),
                    ("anchor", f"{sid}/anchor", False),
                    ("wrong_scene", f"{ids[(ix + 1) % len(ids)]}/A", False),
                    ("prior", None, False),
                    ("feature_shuffle", f"{sid}/feature_shuffle", False),
                    ("channel_permutation", f"{sid}/channel_permutation", False),
                    ("state_mask", f"{sid}/state_mask", False),
                ]
                if residual:
                    methods += [
                        ("Residual_A", f"{sid}/A", True),
                        ("Residual_B", f"{sid}/B", True),
                        ("Residual_shuffled", f"{ids[(ix + 1) % len(ids)]}/A", True),
                        ("Residual_zero", f"{sid}/state_mask", True),
                        ("Residual_channel_permutation", f"{sid}/channel_permutation", True),
                    ]
                reference_prediction = None
                for name, state_key, use_residual in methods:
                    tick = time.perf_counter()
                    if state_key is None:
                        pred = {
                            "rgb": torch.full_like(truth[None, None], 0.5),
                            "depth": torch.full_like(depth[None, None, None], 5.0),
                            "visibility": torch.ones_like(depth[None, None, None]),
                        }
                    else:
                        sealed = states[state_key]
                        before = hash_scene_state(sealed.scene_state)
                        pred = (
                            residual(sealed.scene_state, camera)
                            if use_residual
                            else renderer(
                                sealed.scene_state, camera, ("rgb", "depth", "visibility")
                            )
                        )
                        assert hash_scene_state(sealed.scene_state) == before == sealed.state_hash
                    if device == "cuda":
                        torch.cuda.synchronize()
                    args = (
                        pred["rgb"][0, 0],
                        truth,
                        pred["depth"][0, 0, 0],
                        depth,
                        pred["visibility"][0, 0, 0],
                    )
                    row = {
                        "scene_id": sid,
                        "query_id": fid,
                        "cohort": cohort,
                        "method": name,
                        "split": scene.record["split"],
                        "query_camera_hash": camera_hash,
                        "seconds": time.perf_counter() - tick,
                        **measurement_metrics(*args),
                    }
                    row.update(
                        {
                            region: measurement_metrics(*args, region_mask=mask.cpu().numpy())
                            if mask.any()
                            else None
                            for region, mask in regions.items()
                        }
                    )
                    row["support_bucket_counts"] = {
                        region: int(mask.sum()) for region, mask in regions.items()
                    }
                    if name == "A":
                        reference_prediction = {k: pred[k].clone() for k in ("rgb", "depth")}
                    row["prediction_change_from_A"] = {
                        k: {
                            "rms": float(
                                (pred[k] - reference_prediction[k]).square().mean().sqrt()
                            ),
                            "max_abs": float((pred[k] - reference_prediction[k]).abs().max()),
                        }
                        for k in ("rgb", "depth")
                    }
                    row["prediction_hash"] = hash_value(
                        {"rgb": pred["rgb"], "depth": pred["depth"]}
                    )
                    rows.append(row)
                    assert camera_hash == hash_value(camera)
    ledger.assert_ready()
    write_json(output / "static_results.json", rows)
    write_json(output / "bootstrap_results.json", summarize(rows))
    write_json(output / "sealed_query_access.json", ledger.events)
    assert sha(carrier_path) == metadata["sha256"]
    assert all(sha(Path(p)) == h for p, h in input_hashes.items())
    assert all(sha(Path(p)) == h for p, h in source_hashes.items())
    write_json(
        output / "integrity.json",
        {
            "status": "PASS",
            "checkpoint_unchanged": True,
            "inputs_unchanged": True,
            "sources_unchanged": True,
            "states_sealed_before_query": True,
            "dynamic_ttt_run": False,
            "writes": 0,
            "n_scenes": len(scenes),
            "rows": len(rows),
            "elapsed_seconds": time.perf_counter() - start,
            "residual_parameters": sum(p.numel() for p in residual.head.parameters())
            if residual
            else 0,
        },
    )
    return rows
