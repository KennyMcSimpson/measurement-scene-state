"""Core B V6: does the test-time write need new frames? Context-only TTT next to stream TTT.

PLAN_BEFORE_COREB_V6.md fixed this round. On top of Core B V4/V5's NO_STREAM, OFF and TTT, two
policies supervise the same 10 Adam steps with the 3 warmup context frames instead of the 4 stream
frames: CTX_TTT (no stream cached) and OFF_CTX_TTT (stream cached). TTT is Core B V4/V5's policy
unchanged (same generator); the new policies use generators seeded by (seed, episode, policy).
Everything is sealed before any query (Core B V5's CachedSealBarrier); RGB is clamped to [0, 1].
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import CapacityEvaluator
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_eval_v5 import (
    CachedSealBarrier,
    context_bounds,
    eval_context,
)
from mcss.mechanism_pilot.rgbd_stream_ttt_evaluation_v4 import clamped
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    StreamRGBDLoader,
    cached_episode,
    episode_generator,
    rebuild_with,
    ttt_fast,
)
from mcss.mechanism_pilot.small_training import sha, write_json

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6"
POLICIES = ("NO_STREAM", "CTX_TTT", "OFF", "OFF_CTX_TTT", "TTT")
CONTROLS = {"TTT_WRONG_SCENE": "TTT", "CTX_TTT_WRONG_SCENE": "CTX_TTT"}
WITH_STREAM = {"NO_STREAM": False, "CTX_TTT": False, "OFF": True, "OFF_CTX_TTT": True, "TTT": True}


def generator_for(seed, episode_id, policy):
    """TTT keeps Core B V4/V5's generator; the context policies get their own."""
    if policy == "TTT":
        return episode_generator(seed, episode_id)
    return episode_generator(seed, f"{episode_id}:{policy}")


def unroll_v6(carrier, warmup, stream, bounds, policy, episode_id, settings, seed):
    """(final state, anchor, fast weights, episode, TTT losses) of one fixed-policy episode."""
    if policy not in POLICIES:
        raise ValueError(f"Unregistered policy {policy}")
    episode = cached_episode(carrier, warmup, stream, bounds, episode_id, WITH_STREAM[policy])
    if policy in ("NO_STREAM", "OFF"):
        fast, losses = carrier.initial_fast(episode_id), []
    else:
        supervision = stream if policy == "TTT" else warmup
        fast, losses = ttt_fast(
            carrier,
            episode,
            supervision,
            settings["steps"],
            settings["learning_rate"],
            generator_for(seed, episode_id, policy),
        )
    with torch.no_grad():
        state = episode.run(fast, "materialize")
    return state, episode.anchor.clone(), fast, episode, losses


def cohort_records(cohorts, expected_split):
    """{scene_id: (cohort, record, manifest, base)}: disjoint, unprotected, expected split."""
    records = {}
    for name, (manifest, base) in cohorts.items():
        protected = set(manifest.get("protected_scene_ids", []))
        for record in manifest["scenes"]:
            sid = record["scene_id"]
            if record["split"] != expected_split[name] or sid in protected:
                raise PermissionError(f"{sid}: not an unprotected {name} record")
            if sid in records:
                raise PermissionError(f"{sid} appears in two cohorts")
            records[sid] = (name, record, manifest, Path(base))
    if len(records) < 2:
        raise PermissionError("At least two scenes required for the wrong-scene controls")
    return records


@torch.no_grad()
def evaluate_v6(carrier, settings, cohorts, expected_split, out, seed, device):
    """Every policy and control on every cohort scene, all sealed before any query."""
    out = Path(out)
    if out.exists():
        raise FileExistsError("Refusing to overwrite prior Core B evaluation")
    entries = cohort_records(cohorts, expected_split)
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
                state, anchor, fast, episode, losses = unroll_v6(
                    carrier, warmup, stream, bounds, policy, episode_id, settings, seed
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
                    "stream_frame_ids": [o.frame_id for o in stream] if WITH_STREAM[policy] else [],
                    "supervised_frame_ids": (
                        [o.frame_id for o in stream]
                        if policy == "TTT"
                        else [o.frame_id for o in warmup]
                        if losses
                        else []
                    ),
                    "fast_step": fast.step,
                    "fast_norm": float(fast.delta_fuse.norm() + fast.delta_complete.norm()),
                    "ttt_loss_first": losses[0] if losses else None,
                    "ttt_loss_last": losses[-1] if losses else None,
                    "construction_seconds": time.perf_counter() - tick,
                }
                if policy in CONTROLS.values():
                    episodes[sid, role, policy] = (episode, fast, anchor)
    ids = sorted(entries)
    for sid in ids:
        donor = ids[(ids.index(sid) + 1) % len(ids)]
        for role in ("A", "B"):
            for control, policy in CONTROLS.items():
                episode, _, anchor = episodes[sid, role, policy]
                state = rebuild_with(episode, episodes[donor, role, policy][1])
                key = f"{sid}/{role}/{control}"
                barrier.seal(key, sid, state.to(device="cpu"))
                metadata[key] = {
                    "cohort": entries[sid][0],
                    "scene_id": sid,
                    "role": role,
                    "policy": control,
                    "donor_scene_id": donor,
                    "donor_policy": policy,
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
