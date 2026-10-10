"""Core B V4: the TRAIN-only hyperparameter selection and the sealed DEV evaluation of TTT.

Selection: on the locked TRAIN episodes, every grid setting is scored by AbsRel(OFF) - AbsRel(TTT)
on the episodes' primary queries (offline TRAIN labels), averaged over carriers and episodes. DEV:
every policy state is sealed before any query camera or GT (the Core B V3 barrier and audit); RGB
is clamped to [0, 1] before saving and scoring, and the excess is recorded (as in Core B V3).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import CapacityEvaluator, StateSealBarrier
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.geometry_carrier_experiment import validate_split
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    CONTROLS,
    POLICIES,
    StreamRGBDLoader,
    episode_generator,
    rebuild_with,
    unroll_ttt,
)
from mcss.mechanism_pilot.small_training import sha, write_json

SELECTION_SEED = "COREB-V4-20261003"


def clamped(pred):
    rgb = pred["rgb"].clamp(0, 1)
    return {**pred, "rgb": rgb}, float((pred["rgb"] - rgb).abs().max())


def selection_episodes(eligible, count):
    """The first `count` eligible TRAIN episodes ("scene/role") in sha256(seed:episode) order."""
    key = lambda e: hashlib.sha256(f"{SELECTION_SEED}:{e}".encode()).hexdigest()  # noqa: E731
    return [tuple(e.split("/")) for e in sorted(eligible, key=key)[:count]]


def choose(gains, grid):
    """The largest mean gain; ties go to fewer steps, then the smaller learning rate."""
    order = sorted(
        range(len(grid)),
        key=lambda i: (-round(gains[i], 12), grid[i]["steps"], grid[i]["learning_rate"]),
    )
    return order[0]


@torch.no_grad()
def select_on_train(carriers, manifest, root, episodes, grid, device):
    """{rows, gains, selected, settings} of the TRAIN-only grid selection."""
    validate_split(manifest)
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "TRAIN"}
    holdout = set(manifest.get("protected_scene_ids", [])) | {
        r["scene_id"] for r in manifest["scenes"] if r["split"] != "TRAIN"
    }
    access, data, streams = [], {}, {}
    for sid in sorted({s for s, _ in episodes}):
        data[sid] = DepthBoundsSceneData(records[sid], manifest, root, device, access)
        streams[sid] = StreamRGBDLoader(
            records[sid],
            manifest["image_size"],
            root,
            device,
            access,
            allowed_scene_ids=tuple(records),
            holdout_scene_ids=tuple(holdout),
        )
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    rows = []
    for seed, carrier in carriers.items():
        for sid, role in episodes:
            scene = data[sid]
            warmup, stream = scene.context(role), streams[sid].stream(role)
            episode_id = f"select:{sid}:{role}"
            states = {}
            states["OFF"], anchor, _, _, _ = unroll_ttt(
                carrier, warmup, stream, scene.bounds[role], "OFF", episode_id, None, None
            )
            for index, settings in enumerate(grid):
                generator = episode_generator(seed, episode_id)
                states[f"TTT_{index}"], _, _, _, losses = unroll_ttt(
                    carrier,
                    warmup,
                    stream,
                    scene.bounds[role],
                    "TTT",
                    episode_id,
                    settings,
                    generator,
                )
                rows.append(
                    {
                        "seed": seed,
                        "scene_id": sid,
                        "role": role,
                        "method": f"TTT_{index}",
                        "stream_loss_first": losses[0],
                        "stream_loss_last": losses[-1],
                    }
                )
            for slot in range(len(scene.record["roles"]["primary_query"])):
                target = scene.target(slot)  # offline TRAIN label
                camera = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                for name, state in states.items():
                    pred, _ = clamped(renderer(state, camera, ("rgb", "depth", "visibility")))
                    rows.append(
                        {
                            "seed": seed,
                            "scene_id": sid,
                            "role": role,
                            "slot": slot,
                            "method": name,
                            "depth_absrel": _metrics(pred, target.rgb, target.depth)[
                                "depth_absrel"
                            ],
                        }
                    )
    scored = [r for r in rows if "depth_absrel" in r]
    base = {
        (r["seed"], r["scene_id"], r["role"], r["slot"]): r["depth_absrel"]
        for r in scored
        if r["method"] == "OFF"
    }
    gains = {}
    for index in range(len(grid)):
        values = [
            base[r["seed"], r["scene_id"], r["role"], r["slot"]] - r["depth_absrel"]
            for r in scored
            if r["method"] == f"TTT_{index}"
        ]
        gains[index] = float(np.mean(values))
    selected = choose(gains, grid)
    return {
        "rows": rows,
        "gains": {str(k): v for k, v in gains.items()},
        "selected": selected,
        "settings": grid[selected],
        "episodes": [f"{s}/{r}" for s, r in episodes],
        "access": access,
    }


@torch.no_grad()
def evaluate_ttt(carrier, settings, manifest, root, out, seed, device):
    """DEV only: NO_STREAM, OFF, TTT and TTT_WRONG_SCENE, every state sealed before any query."""
    validate_split(manifest)
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "DEV"}
    if not records:
        raise PermissionError("DEV scenes required")
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError("Refusing to overwrite prior Core B evaluation")
    forbidden = set(manifest.get("protected_scene_ids", [])) | {
        r["scene_id"] for r in manifest["scenes"] if r["split"] != "DEV"
    }
    names = (*POLICIES, *CONTROLS)
    keys = [f"{sid}/{role}/{name}" for sid in records for role in ("A", "B") for name in names]
    barrier = StateSealBarrier(keys)
    access, metadata, episodes = [], {}, {}
    before = hash_value(carrier.state_dict())
    for sid, record in records.items():
        data = DepthBoundsSceneData(record, manifest, root, device, access)
        streams = StreamRGBDLoader(
            record,
            manifest["image_size"],
            root,
            device,
            access,
            allowed_scene_ids=tuple(records),
            holdout_scene_ids=tuple(forbidden),
        )
        for role in ("A", "B"):
            if streams.frame_ids(role) is None:
                raise PermissionError(f"DEV episode {sid}/{role} lacks a full stream")
            warmup, stream = data.context(role), streams.stream(role)
            episode_id = f"dev:{sid}:{role}"
            for policy in POLICIES:
                tick = time.perf_counter()
                generator = episode_generator(seed, episode_id) if policy == "TTT" else None
                state, anchor, fast, episode, losses = unroll_ttt(
                    carrier,
                    warmup,
                    stream,
                    data.bounds[role],
                    policy,
                    episode_id,
                    settings,
                    generator,
                )
                key = f"{sid}/{role}/{policy}"
                barrier.seal(key, sid, state.to(device="cpu"))
                metadata[key] = {
                    "scene_id": sid,
                    "role": role,
                    "policy": policy,
                    "anchor_c2w": anchor.cpu().tolist(),
                    "state_hash": hash_scene_state(state),
                    "bounds": state.bounds.cpu().tolist(),
                    "warmup_frame_ids": [o.frame_id for o in warmup],
                    "stream_frame_ids": []
                    if policy == "NO_STREAM"
                    else [o.frame_id for o in stream],
                    "fast_step": fast.step,
                    "fast_norm": float(fast.delta_fuse.norm() + fast.delta_complete.norm()),
                    "ttt_stream_loss_first": losses[0] if losses else None,
                    "ttt_stream_loss_last": losses[-1] if losses else None,
                    "construction_seconds": time.perf_counter() - tick,
                }
                if policy == "TTT":
                    episodes[sid, role] = (episode, fast, anchor)
    ids = sorted(records)
    for sid in ids:
        donor = ids[(ids.index(sid) + 1) % len(ids)]
        if donor == sid:
            raise PermissionError("At least two DEV scenes required for the wrong-scene control")
        for role in ("A", "B"):
            episode, _, anchor = episodes[sid, role]
            state = rebuild_with(episode, episodes[donor, role][1])
            key = f"{sid}/{role}/TTT_WRONG_SCENE"
            barrier.seal(key, sid, state.to(device="cpu"))
            metadata[key] = {
                "scene_id": sid,
                "role": role,
                "policy": "TTT_WRONG_SCENE",
                "donor_scene_id": donor,
                "anchor_c2w": anchor.cpu().tolist(),
                "state_hash": hash_scene_state(state),
                "bounds": state.bounds.cpu().tolist(),
            }
    barrier.assert_ready()
    preseal_depth_audit(access, records)
    seal_boundary = len(access)
    access.append({"event": "ALL_STREAM_STATES_SEALED", "state_count": len(keys)})
    out.mkdir(parents=True)
    torch.save({key: barrier.state(key) for key in keys}, out / "states.pt")
    write_json(out / "state_hashes.json", metadata)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048).to(device)
    (out / "predictions").mkdir()
    rows = []
    for sid, record in records.items():
        evaluator = CapacityEvaluator(
            record,
            manifest["image_size"],
            root,
            device,
            barrier=barrier,
            capacity_scene_ids=tuple(records),
            holdout_scene_ids=tuple(forbidden),
            access_log=access,
        )
        for role in ("A", "B"):
            anchor = torch.tensor(
                metadata[f"{sid}/{role}/NO_STREAM"]["anchor_c2w"],
                dtype=torch.float32,
                device=device,
            )
            for qid in record["roles"]["primary_query"]:
                batch = evaluator.query(qid)
                camera = transform_cameras(batch.cameras, torch.linalg.inv(anchor))
                for name in names:
                    key = f"{sid}/{role}/{name}"
                    state = barrier.state(key).to(device=device)
                    pred, excess = clamped(renderer(state, camera, ("rgb", "depth", "visibility")))
                    path = out / "predictions" / f"{sid}_{role}_{qid}_{name}.npz"
                    np.savez_compressed(
                        path,
                        rgb=pred["rgb"][0, 0].cpu().numpy(),
                        depth=pred["depth"][0, 0, 0].cpu().numpy(),
                        opacity=pred["visibility"][0, 0, 0].cpu().numpy(),
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "scene_id": sid,
                            "role": role,
                            "query_id": qid,
                            "policy": name,
                            "used_state_hash": hash_scene_state(state),
                            "prediction_path": str(path.relative_to(out)),
                            "prediction_file_sha256": sha(path),
                            "rgb_clamp_max_excess": excess,
                            **_metrics(pred, batch.rgb, batch.depth),
                        }
                    )
    if hash_value(carrier.state_dict()) != before:
        raise AssertionError("Evaluation changed a carrier tensor")
    for name, value in (
        ("query_results", rows),
        ("GT_access", access),
        ("seal_events", barrier.events),
    ):
        write_json(out / f"{name}.json", value)
    write_json(
        out / "integrity.json",
        {
            "status": "PASS",
            "seed": seed,
            "settings": settings,
            "states": len(keys),
            "query_rows": len(rows),
            "states_file_sha256": sha(out / "states.pt"),
            "all_states_sealed_before_GT": True,
            "preseal_access_count": seal_boundary,
            "slow_carrier_unchanged": True,
            "test_time_input": "RGB+DEPTH+CAMERA of warmup and stream frames",
        },
    )
    return rows
