"""Core B V1 DEV evaluation: every policy state is sealed before any query camera or GT.

Before sealing, depth is read only for warmup context frames (CONTEXT_ONLY_RGBD, via the V7
context-only loader) and for stream frames (STREAM_RGBD); no query frame is reachable.
"""

from __future__ import annotations

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
from mcss.mechanism_pilot.rgbd_stream_write import (
    CONTROLS,
    POLICIES,
    StreamRGBDLoader,
    rebuild_with,
    unroll,
)
from mcss.mechanism_pilot.small_training import sha, write_json

PRESEAL_DEPTH_PURPOSES = {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}


def preseal_depth_audit(access, records):
    """Before the seal marker: depth only for context or stream frames, never for a query."""
    for event in access:
        if "depth" not in event.get("channels", []):
            continue
        roles = records[event["scene_id"]]["roles"]
        queries = set(roles["primary_query"]) | set(roles.get("query", ()))
        if event.get("purpose") not in PRESEAL_DEPTH_PURPOSES or event.get("frame_id") in queries:
            raise AssertionError("Non-context/non-stream depth accessed before state sealing")


@torch.no_grad()
def evaluate_streams(carrier, rules, manifest, root, out, seed, device):
    """DEV only. `rules` maps "trained" / "untrained" to write rules."""
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
    before = {k: hash_value(v.state_dict()) for k, v in {"carrier": carrier, **rules}.items()}
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
            for policy in POLICIES:
                rule = rules["untrained"] if policy == "ALL_UNTRAINED" else rules["trained"]
                tick = time.perf_counter()
                state, anchor, fast, episode = unroll(
                    carrier, rule, warmup, stream, data.bounds[role], policy, f"dev:{sid}:{role}"
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
                    "construction_seconds": time.perf_counter() - tick,
                }
                if policy == "ALL":
                    episodes[sid, role] = (episode, fast, anchor)
    ids = sorted(records)
    for sid in ids:
        donor = ids[(ids.index(sid) + 1) % len(ids)]
        if donor == sid:
            raise PermissionError("At least two DEV scenes required for the wrong-scene control")
        for role in ("A", "B"):
            episode, _, anchor = episodes[sid, role]
            state = rebuild_with(episode, episodes[donor, role][1])
            key = f"{sid}/{role}/ALL_WRONG_SCENE"
            barrier.seal(key, sid, state.to(device="cpu"))
            metadata[key] = {
                "scene_id": sid,
                "role": role,
                "policy": "ALL_WRONG_SCENE",
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
                    pred = renderer(state, camera, ("rgb", "depth", "visibility"))
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
                            **_metrics(pred, batch.rgb, batch.depth),
                        }
                    )
    after = {k: hash_value(v.state_dict()) for k, v in {"carrier": carrier, **rules}.items()}
    if after != before:
        raise AssertionError("Evaluation changed a carrier or write-rule tensor")
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
            "states": len(keys),
            "query_rows": len(rows),
            "states_file_sha256": sha(out / "states.pt"),
            "all_states_sealed_before_GT": True,
            "preseal_access_count": seal_boundary,
            "preseal_depth_purposes": sorted(PRESEAL_DEPTH_PURPOSES),
            "slow_carrier_and_rules_unchanged": True,
            "test_time_input": "RGB+DEPTH+CAMERA of warmup and stream frames",
        },
    )
    return rows
