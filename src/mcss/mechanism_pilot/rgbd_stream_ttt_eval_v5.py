"""Core B V5: Core B V4's TTT, unchanged, on the independent EVAL-V3 + EVAL-V4 cohorts (sealed).

PLAN_BEFORE_COREB_V5.md fixed this round. The V4 setting (selected on TRAIN) is applied without any
new selection. Episodes follow Core B V1-V4: one context role warms the state up, with bounds from
that role's measured context depth (DepthBoundsSceneData's rule), then the 4 stream frames of the
frame-id rule arrive. The EVAL records never pass through the TRAIN/DEV scene-data classes: their
contexts come from the same context-only RGB-D loader with the cohort's own scene ids. Every policy
state is sealed before any query camera or GT (the Core B barrier and pre-seal depth audit); RGB is
clamped to [0, 1] before saving and scoring, and the excess is recorded (as in Core B V3/V4).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import OnlineObservation, hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import (
    CapacityEvaluator,
    ContextOnlyLoader,
    StateSealBarrier,
)
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData, context_depth_bounds
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_evaluation_v4 import clamped
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    CONTROLS,
    POLICIES,
    StreamRGBDLoader,
    episode_generator,
    rebuild_with,
    unroll_ttt,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-EVAL-V5"
COHORTS = ("EVAL_V3", "EVAL_V4")


class CachedSealBarrier(StateSealBarrier):
    """StateSealBarrier whose full integrity re-hash runs when the seal closes and at release.

    StateSealBarrier.assert_ready re-hashes every sealed state on each state() and read() call,
    which is quadratic in the number of states (664 per seed here, about 3 GB). The seal is kept:
    nothing is readable before every planned state is sealed and verified, nothing can be sealed
    afterwards, and release() re-verifies every hash after the last query.
    """

    def __init__(self, expected_state_ids):
        super().__init__(expected_state_ids)
        self._verified = False

    def seal(self, state_id, scene_id, state):
        if self._verified:
            raise PermissionError("The seal is closed")
        super().seal(state_id, scene_id, state)

    def assert_ready(self):
        if not self._verified:
            super().assert_ready()
            self._verified = True

    def release(self):
        """StateSealBarrier's full check, after every query."""
        StateSealBarrier.assert_ready(self)


def eval_context(record, manifest, base, device, access):
    """{role: RGBDContext}: one scene's context RGB, measured depth and cameras, nothing else."""
    loader = ContextOnlyLoader(
        record,
        manifest["image_size"],
        base,
        device,
        track="RGBD",
        capacity_scene_ids=[r["scene_id"] for r in manifest["scenes"]],
        holdout_scene_ids=manifest.get("protected_scene_ids", []),
        access_log=access,
    )
    contexts = {}
    for role in ("A", "B"):
        batch = loader.context(role)
        observations = [
            OnlineObservation(
                batch.scene_id,
                fid,
                batch.rgb[0, j],
                Cameras(
                    batch.cameras.intrinsics[0, j],
                    batch.cameras.c2w[0, j],
                    batch.cameras.image_size,
                ),
            )
            for j, fid in enumerate(batch.frame_ids)
        ]
        contexts[role] = RGBDContext(observations, batch.depth[0, :, 0])
    return contexts


def context_bounds(context, device):
    """DepthBoundsSceneData's rule: the role's context-depth hull in its anchor frame."""
    anchor = context[0].camera.c2w
    return context_depth_bounds(context, anchor, DepthBoundsSceneData.TRIM).to(device)


def cohort_records(cohorts):
    """{scene_id: (cohort, record, manifest, base)}; cohorts must be disjoint and unprotected."""
    records = {}
    for name, (manifest, base) in cohorts.items():
        protected = set(manifest.get("protected_scene_ids", []))
        for record in manifest["scenes"]:
            sid = record["scene_id"]
            if record["split"] != name or sid in protected:
                raise PermissionError(f"{sid}: not an unprotected {name} record")
            if sid in records:
                raise PermissionError(f"{sid} appears in two cohorts")
            records[sid] = (name, record, manifest, Path(base))
    if len(records) < 2:
        raise PermissionError("At least two scenes required for the wrong-scene control")
    return records


@torch.no_grad()
def evaluate_ttt_cohorts(carrier, settings, cohorts, out, seed, device):
    """NO_STREAM, OFF, TTT and TTT_WRONG_SCENE on every cohort scene, all sealed before any query.

    `cohorts` maps a cohort name to (manifest, base directory of its relative paths).
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError("Refusing to overwrite prior Core B evaluation")
    entries = cohort_records(cohorts)
    records = {sid: item[1] for sid, item in entries.items()}
    names = (*POLICIES, *CONTROLS)
    keys = [f"{sid}/{role}/{name}" for sid in entries for role in ("A", "B") for name in names]
    barrier = CachedSealBarrier(keys)
    access, metadata, episodes = [], {}, {}
    before = hash_value(carrier.state_dict())
    for sid, (cohort, record, manifest, base) in entries.items():
        ids = tuple(r["scene_id"] for r in manifest["scenes"])
        protected = tuple(manifest.get("protected_scene_ids", []))
        contexts = eval_context(record, manifest, base, device, access)
        streams = StreamRGBDLoader(
            record,
            manifest["image_size"],
            base,
            device,
            access,
            allowed_scene_ids=ids,
            holdout_scene_ids=protected,
        )
        for role in ("A", "B"):
            if streams.frame_ids(role) is None:
                raise PermissionError(f"{cohort} episode {sid}/{role} lacks a full stream")
            warmup, stream = contexts[role], streams.stream(role)
            bounds = context_bounds(warmup, device)
            episode_id = f"{cohort.lower()}:{sid}:{role}"
            for policy in POLICIES:
                tick = time.perf_counter()
                generator = episode_generator(seed, episode_id) if policy == "TTT" else None
                state, anchor, fast, episode, losses = unroll_ttt(
                    carrier, warmup, stream, bounds, policy, episode_id, settings, generator
                )
                key = f"{sid}/{role}/{policy}"
                barrier.seal(key, sid, state.to(device="cpu"))
                metadata[key] = {
                    "cohort": cohort,
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
    ids = sorted(entries)
    for sid in ids:
        donor = ids[(ids.index(sid) + 1) % len(ids)]
        for role in ("A", "B"):
            episode, _, anchor = episodes[sid, role]
            state = rebuild_with(episode, episodes[donor, role][1])
            key = f"{sid}/{role}/TTT_WRONG_SCENE"
            barrier.seal(key, sid, state.to(device="cpu"))
            metadata[key] = {
                "cohort": entries[sid][0],
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
    for sid, (cohort, record, manifest, base) in entries.items():
        evaluator = CapacityEvaluator(
            record,
            manifest["image_size"],
            base,
            device,
            barrier=barrier,
            capacity_scene_ids=tuple(r["scene_id"] for r in manifest["scenes"]),
            holdout_scene_ids=tuple(manifest.get("protected_scene_ids", [])),
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
                            "cohort": cohort,
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
    barrier.release()
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
            "cohorts": sorted(cohorts),
            "states": len(keys),
            "query_rows": len(rows),
            "states_file_sha256": sha(out / "states.pt"),
            "all_states_sealed_before_GT": True,
            "seal_barrier": "CachedSealBarrier: full re-hash when the seal closes and at release",
            "preseal_access_count": seal_boundary,
            "slow_carrier_unchanged": True,
            "test_time_input": "RGB+DEPTH+CAMERA of warmup and stream frames",
        },
    )
    return rows
