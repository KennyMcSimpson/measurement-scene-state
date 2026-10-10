"""Core B V7: stream and context-only TTT on the fresh EVAL-V5 cohort, policies from one table.

Each policy is (stream cached?, supervised frames, Adam steps) at learning rate 1e-3. TTT keeps
Core B V4/V5's generator (seeded by (seed, episode)); every other TTT policy is seeded by (seed,
episode, policy), which for CTX_TTT and OFF_CTX_TTT is Core B V6's rule. Episodes, the seal
(Core B V5's CachedSealBarrier), rendering and scoring are Core B V6's.
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
from mcss.mechanism_pilot.rgbd_stream_ttt_context_v6 import cohort_records
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

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7"
LEARNING_RATE = 1e-3
POLICY_SPECS = {
    "NO_STREAM": (False, None, 0),
    "CTX_TTT": (False, "context", 10),
    "CTX_TTT_100": (False, "context", 100),
    "OFF": (True, None, 0),
    "OFF_CTX_TTT": (True, "context", 10),
    "OFF_CTX_TTT_100": (True, "context", 100),
    "TTT": (True, "stream", 10),
    "TTT_100": (True, "stream", 100),
}
CONTROLS = {
    "TTT_WRONG_SCENE": "TTT",
    "TTT_100_WRONG_SCENE": "TTT_100",
    "CTX_TTT_WRONG_SCENE": "CTX_TTT",
    "CTX_TTT_100_WRONG_SCENE": "CTX_TTT_100",
}


def generator_for(seed, episode_id, policy):
    """TTT keeps Core B V4/V5's generator; every other policy gets its own."""
    if policy == "TTT":
        return episode_generator(seed, episode_id)
    return episode_generator(seed, f"{episode_id}:{policy}")


def unroll_v7(carrier, warmup, stream, bounds, policy, episode_id, seed, specs=None):
    """(final state, anchor, fast weights, episode, TTT losses) of one fixed-policy episode."""
    specs = POLICY_SPECS if specs is None else specs
    if policy not in specs:
        raise ValueError(f"Unregistered policy {policy}")
    with_stream, supervision, steps = specs[policy]
    episode = cached_episode(carrier, warmup, stream, bounds, episode_id, with_stream)
    if steps == 0:
        fast, losses = carrier.initial_fast(episode_id), []
    else:
        frames = stream if supervision == "stream" else warmup
        generator = generator_for(seed, episode_id, policy)
        fast, losses = ttt_fast(carrier, episode, frames, steps, LEARNING_RATE, generator)
    with torch.no_grad():
        state = episode.run(fast, "materialize")
    return state, episode.anchor.clone(), fast, episode, losses


@torch.no_grad()
def evaluate_v7(carrier, cohorts, expected_split, out, seed, device, specs=None):
    """Every policy and control on every cohort scene, all sealed before any query."""
    specs = POLICY_SPECS if specs is None else specs
    out = Path(out)
    if out.exists():
        raise FileExistsError("Refusing to overwrite prior Core B evaluation")
    entries = cohort_records(cohorts, expected_split)
    records = {sid: item[1] for sid, item in entries.items()}
    names = (*specs, *CONTROLS)
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
            for policy, (with_stream, supervision, steps) in specs.items():
                tick = time.perf_counter()
                state, anchor, fast, episode, losses = unroll_v7(
                    carrier, warmup, stream, bounds, policy, episode_id, seed, specs
                )
                key = f"{sid}/{role}/{policy}"
                barrier.seal(key, sid, state.to(device="cpu"))
                supervised = {"stream": stream, "context": warmup}.get(supervision, [])
                metadata[key] = {
                    "cohort": cohort,
                    "scene_id": sid,
                    "role": role,
                    "policy": policy,
                    "anchor_c2w": anchor.cpu().tolist(),
                    "state_hash": hash_scene_state(state),
                    "bounds": state.bounds.cpu().tolist(),
                    "warmup_frame_ids": [o.frame_id for o in warmup],
                    "stream_frame_ids": [o.frame_id for o in stream] if with_stream else [],
                    "supervised_frame_ids": [o.frame_id for o in supervised],
                    "ttt_steps": steps,
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
            "policies": {k: list(v) for k, v in specs.items()},
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
