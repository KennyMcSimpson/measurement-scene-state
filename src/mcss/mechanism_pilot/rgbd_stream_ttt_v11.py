"""Core B V11: online stream writes (one write per arriving frame) against the one-shot write.

PLAN_BEFORE_COREB_V11.md fixed this round. NO_STREAM, OFF and TTT_100 are Core B V7's rows and
are unrolled by Core B V7's `unroll_v7`. ONLINE policies cache the warmup only; each stream frame
then arrives, is cached, and the fast matrices take steps/4 Adam steps against the frames seen so
far (ONLINE) or the newest frame only (ONLINE_NEW), with one optimizer for the whole episode.
ALL_TTT_100 is the one-shot write supervised by the context and the stream frames together. Every
policy is seeded by (seed, episode, policy) as in Core B V7. The evaluation is Core B V8's with
this round's unroll.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import FastWeights, hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_contracts import CapacityEvaluator
from mcss.mechanism_pilot.direct_capacity_evaluation import _metrics
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.mechanism_pilot.rgbd_stream_evaluation_v3 import preseal_depth_audit
from mcss.mechanism_pilot.rgbd_stream_ttt_context_v6 import cohort_records
from mcss.mechanism_pilot.rgbd_stream_ttt_eval_v5 import (
    CachedSealBarrier,
    context_bounds,
    eval_context,
)
from mcss.mechanism_pilot.rgbd_stream_ttt_evaluation_v4 import clamped
from mcss.mechanism_pilot.rgbd_stream_ttt_v4 import (
    RAYS_PER_FRAME,
    StreamRGBDLoader,
    cached_episode,
    rebuild_with,
    stream_targets,
    training_loss,
    ttt_fast,
)
from mcss.mechanism_pilot.rgbd_stream_ttt_v7 import LEARNING_RATE, generator_for, unroll_v7
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11"
POLICY_SPECS = {
    "NO_STREAM": (False, None, 0),
    "OFF": (True, None, 0),
    "TTT_100": (True, "stream", 100),
    "ONLINE_100": (True, "online", 100),
    "ONLINE_NEW_100": (True, "online_new", 100),
    "ALL_TTT_100": (True, "all", 100),
}
CONTROLS = {"ONLINE_100_WRONG_SCENE": "ONLINE_100"}
__all__ = [
    "CONTROLS",
    "EXPERIMENT",
    "LEARNING_RATE",
    "POLICY_SPECS",
    "evaluate_v11",
    "ttt_online",
    "unroll_v11",
]


def ttt_online(carrier, episode, stream, steps_per_frame, learning_rate, generator, replay):
    """Stream frames arrive one by one; after each is cached, Adam steps on the fast matrices.

    The matrices start at zero and one optimizer carries its moments across arrivals. Each step
    samples 256 rays per supervised frame: every frame seen so far (replay) or the newest only.
    """
    device = carrier._bounds.device
    zero = carrier.initial_fast(episode.cache.episode_id)
    delta_fuse = zero.delta_fuse.clone().requires_grad_(True)
    delta_complete = zero.delta_complete.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([delta_fuse, delta_complete], lr=learning_rate)
    observations, losses = list(stream), []
    for k, (observation, depth) in enumerate(zip(stream, stream.depths, strict=True)):
        episode.arrive(observation, depth)
        first = 0 if replay else k
        frames = RGBDContext(observations[first : k + 1], stream.depths[first : k + 1])
        local, rgb, target = stream_targets(frames, episode.anchor)
        local = Cameras(local.intrinsics.to(device), local.c2w.to(device), local.image_size)
        rgb, target = rgb.to(device), target.to(device, torch.float32)
        views = rgb.shape[1]
        height, width = local.image_size
        with torch.enable_grad():
            for _ in range(steps_per_frame):
                fast = FastWeights(episode.cache.episode_id, delta_fuse, delta_complete)
                state = episode.run(fast, "materialize")
                offsets = torch.arange(views).repeat_interleave(RAYS_PER_FRAME) * (height * width)
                rays = torch.randint(height * width, (views * RAYS_PER_FRAME,), generator=generator)
                loss, _ = training_loss(state, local, rgb, target, (offsets + rays).to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite test-time objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
    final = FastWeights(
        episode.cache.episode_id,
        delta_fuse.detach().clone(),
        delta_complete.detach().clone(),
        steps_per_frame * len(observations),
    )
    return final, losses


def unroll_v11(carrier, warmup, stream, bounds, policy, episode_id, seed, specs=None):
    """(final state, anchor, fast weights, episode, TTT losses) of one fixed-policy episode."""
    specs = POLICY_SPECS if specs is None else specs
    if policy not in specs:
        raise ValueError(f"Unregistered policy {policy}")
    _, supervision, steps = specs[policy]
    if supervision not in ("online", "online_new", "all"):
        return unroll_v7(carrier, warmup, stream, bounds, policy, episode_id, seed, specs)
    generator = generator_for(seed, episode_id, policy)
    if supervision == "all":
        episode = cached_episode(carrier, warmup, stream, bounds, episode_id, True)
        frames = RGBDContext([*warmup, *stream], torch.cat((warmup.depths, stream.depths)))
        fast, losses = ttt_fast(carrier, episode, frames, steps, LEARNING_RATE, generator)
    else:
        if steps % len(stream):
            raise ValueError("Online steps must split evenly over the stream frames")
        episode = cached_episode(carrier, warmup, stream, bounds, episode_id, False)
        fast, losses = ttt_online(
            carrier,
            episode,
            stream,
            steps // len(stream),
            LEARNING_RATE,
            generator,
            supervision == "online",
        )
    with torch.no_grad():
        state = episode.run(fast, "materialize")
    return state, episode.anchor.clone(), fast, episode, losses


def supervised_frames(supervision, warmup, stream):
    return {
        "stream": list(stream),
        "context": list(warmup),
        "online": list(stream),
        "online_new": list(stream),
        "all": [*warmup, *stream],
    }.get(supervision, [])


@torch.no_grad()
def evaluate_v11(carrier, cohorts, expected_split, out, seed, device, specs=None, controls=None):
    """Every policy and control on every cohort scene, all sealed before any query."""
    specs = POLICY_SPECS if specs is None else specs
    controls = CONTROLS if controls is None else controls
    out = Path(out)
    if out.exists():
        raise FileExistsError("Refusing to overwrite prior Core B evaluation")
    entries = cohort_records(cohorts, expected_split)
    records = {sid: item[1] for sid, item in entries.items()}
    names = (*specs, *controls)
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
                state, anchor, fast, episode, losses = unroll_v11(
                    carrier, warmup, stream, bounds, policy, episode_id, seed, specs
                )
                key = f"{sid}/{role}/{policy}"
                barrier.seal(key, sid, state.to(device="cpu"))
                supervised = supervised_frames(supervision, warmup, stream)
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
                    "cached_frame_ids": list(episode.source_frames),
                    "supervised_frame_ids": [o.frame_id for o in supervised],
                    "supervision": supervision,
                    "ttt_steps": steps,
                    "fast_step": fast.step,
                    "fast_norm": float(fast.delta_fuse.norm() + fast.delta_complete.norm()),
                    "ttt_loss_first": losses[0] if losses else None,
                    "ttt_loss_last": losses[-1] if losses else None,
                    "construction_seconds": time.perf_counter() - tick,
                }
                if policy in controls.values():
                    episodes[sid, role, policy] = (episode, fast, anchor)
    ids = sorted(entries)
    for sid in ids:
        donor = ids[(ids.index(sid) + 1) % len(ids)]
        for role in ("A", "B"):
            for control, policy in controls.items():
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
            "controls": controls,
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
