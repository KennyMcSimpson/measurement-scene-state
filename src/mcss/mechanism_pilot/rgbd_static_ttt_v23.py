"""V23: context-only test-time refinement of the static V17 L state (Core B V6's CTX_TTT).

PLAN_BEFORE_V23.md fixed this round. The static state is one context role's 3 views with bounds
from their measured depth; no stream frame is read. CTX_TTT is Core B V6's policy unchanged (10
Adam steps, learning rate 1e-3, generator seeded by (seed, episode, "CTX_TTT")); CTX_TTT_30 and
CTX_TTT_100 are descriptive dose arms with their own generators. Every state is sealed before any
query camera or GT (Core B V5's CachedSealBarrier); RGB is clamped to [0, 1].
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
    cached_episode,
    episode_generator,
    rebuild_with,
    ttt_fast,
)
from mcss.mechanism_pilot.small_training import sha, write_json

EXPERIMENT = "EXP-3D-RGBD-STATIC-TTT-EVAL-V23"
STEPS = {"NO_STREAM": 0, "CTX_TTT": 10, "CTX_TTT_30": 30, "CTX_TTT_100": 100}
POLICIES = tuple(STEPS)
CONTROLS = {"CTX_TTT_WRONG_SCENE": "CTX_TTT"}
LEARNING_RATE = 1e-3


def unroll_static(carrier, warmup, bounds, policy, episode_id, seed):
    """(final state, anchor, fast weights, episode, TTT losses) of one static episode."""
    if policy not in STEPS:
        raise ValueError(f"Unregistered policy {policy}")
    episode = cached_episode(carrier, warmup, None, bounds, episode_id, False)
    if STEPS[policy] == 0:
        fast, losses = carrier.initial_fast(episode_id), []
    else:
        generator = episode_generator(seed, f"{episode_id}:{policy}")
        fast, losses = ttt_fast(carrier, episode, warmup, STEPS[policy], LEARNING_RATE, generator)
    with torch.no_grad():
        state = episode.run(fast, "materialize")
    return state, episode.anchor.clone(), fast, episode, losses


@torch.no_grad()
def evaluate_static(carrier, cohorts, expected_split, out, seed, device):
    """Every policy and the control on every cohort scene, all sealed before any query."""
    out = Path(out)
    if out.exists():
        raise FileExistsError("Refusing to overwrite prior V23 evaluation")
    entries = cohort_records(cohorts, expected_split)
    records = {sid: item[1] for sid, item in entries.items()}
    names = (*POLICIES, *CONTROLS)
    keys = [f"{sid}/{role}/{name}" for sid in entries for role in ("A", "B") for name in names]
    barrier = CachedSealBarrier(keys)
    access, metadata, episodes = [], {}, {}
    before = hash_value(carrier.state_dict())
    for sid, (cohort, record, manifest, base) in entries.items():
        contexts = eval_context(record, manifest, base, device, access)
        for role in ("A", "B"):
            warmup = contexts[role]
            bounds = context_bounds(warmup, device)
            episode_id = f"{cohort.lower()}:{sid}:{role}"
            for policy in POLICIES:
                tick = time.perf_counter()
                state, anchor, fast, episode, losses = unroll_static(
                    carrier, warmup, bounds, policy, episode_id, seed
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
                    "ttt_steps": STEPS[policy],
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
    if any(
        "depth" in e.get("channels", []) and e.get("purpose") != "CONTEXT_ONLY_RGBD" for e in access
    ):
        raise AssertionError("Only context depth may be read before the seal")
    seal_boundary = len(access)
    access.append({"event": "ALL_STATIC_STATES_SEALED", "state_count": len(keys)})
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
            "cohorts": sorted(cohorts),
            "states": len(keys),
            "query_rows": len(rows),
            "states_file_sha256": sha(out / "states.pt"),
            "all_states_sealed_before_GT": True,
            "seal_barrier": "CachedSealBarrier: full re-hash when the seal closes and at release",
            "preseal_access_count": seal_boundary,
            "slow_carrier_unchanged": True,
            "test_time_input": "RGB+DEPTH+CAMERA of the context frames",
        },
    )
    return rows
