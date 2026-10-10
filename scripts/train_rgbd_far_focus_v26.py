#!/usr/bin/env python3
"""V26 far-focus runs of the V17 L recipe; no writer or test cohort access.

Derived from the frozen V17 training script; only the ray sampling differs. Each role draws
its own 1024 rays (plan amendment 1): min(768, FAR pixel count) distinct FAR pixels, drawn
without replacement from the valid query pixels farther than 8 px from the reprojected hits of
every view used to build that role's state (V11 geometry and the INPAINT-V1 NEAR mask, as in
the evaluation), and the rest uniformly over the image; without FAR pixels all are uniform.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.geometry_carrier_experiment import (
    select_checkpoint,
    verify_data,
    verify_lock,
)
from mcss.mechanism_pilot.rgbd_long_train_carrier import (
    EXTRA_SEED_OFFSET,
    VARIANTS,
    build_state,
    extra_count,
    make_carrier,
    parameter_hash,
    save_checkpoint,
    scene_data,
    train_records,
    training_loss,
    with_extra_views,
)
from mcss.mechanism_pilot.rgbd_stream_write import StreamRGBDLoader
from mcss.mechanism_pilot.small_training import AccessLog, sha, write_json
from mcss.types import Cameras

FAR_FRACTION = 0.75
SCRIPTS = Path(__file__).resolve().parent


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


GEOMETRY = _module("v26_geometry", "run_rgbd_geometric_baselines.py")
INPAINT = _module("v26_inpaint", "run_rgbd_inpaint_baselines.py")


def far_pixels(views, target, image_size):
    """Flat indices of valid query pixels farther than 8 px from every measured hit."""
    points = GEOMETRY.context_points(views).cpu()
    intrinsics = target.cameras.intrinsics.reshape(3, 3).cpu()
    c2w = target.cameras.c2w.reshape(4, 4).cpu()
    _, hit = GEOMETRY.reproject(points, Cameras(intrinsics, c2w, tuple(image_size)))
    near = INPAINT.near_mask(hit).reshape(-1)
    depth = target.depth.detach().reshape(-1).cpu().numpy()
    valid = np.isfinite(depth) & (depth > 0)
    return torch.from_numpy(np.flatnonzero(valid & ~near))


def far_focus_indices(far, ray_count, batch_rays, generator):
    """Up to FAR_FRACTION of the rays from distinct FAR pixels, the rest uniform."""
    n_far = min(round(FAR_FRACTION * batch_rays), far.numel())
    picks = far[torch.randperm(far.numel(), generator=generator)[:n_far]] if n_far else far[:0]
    rest = torch.randint(ray_count, (batch_rays - n_far,), generator=generator)
    return torch.cat((picks.long(), rest))


def train(root, variant, seed, device):
    root = Path(root).resolve()
    manifest, config = verify_lock(root)
    verify_data(root)
    torch.set_num_threads(config["num_threads"])
    if variant not in config["variants"] or seed not in config["seeds"]:
        raise PermissionError("Unregistered model/seed")
    destination = root / "checkpoints" / f"{variant}_{seed}"
    destination.mkdir(exist_ok=False)
    access = AccessLog(destination / "access.jsonl")
    records = train_records(manifest, config, variant)
    scenes = [scene_data(variant)(r, manifest, root, device, access) for r in records]
    holdout = set(manifest.get("protected_scene_ids", [])) | {
        r["scene_id"] for r in manifest["scenes"] if r["split"] != "TRAIN"
    }
    streams = [
        StreamRGBDLoader(
            r,
            manifest["image_size"],
            root,
            device,
            access,
            allowed_scene_ids=[x["scene_id"] for x in records],
            holdout_scene_ids=holdout,
        )
        for r in records
    ]
    # Extra-view counts have their own generator: scene order and rays stay matched.
    extra_generator = torch.Generator(device="cpu").manual_seed(seed + EXTRA_SEED_OFFSET)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(scenes), generator=generator).tolist()
    prior = json.loads((root / "train_depth_prior.json").read_text())
    carrier = make_carrier(variant, seed, device, prior["near_m"], prior["far_m"])
    carrier.train()
    initial_hash = hash_value(carrier.state_dict())
    initial_parameters = parameter_hash(carrier)
    optimizer = torch.optim.Adam(carrier.parameters(), lr=config["learning_rate"], weight_decay=0)
    stream = hashlib.sha256()
    candidates = []
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with (destination / "training.jsonl").open("x", buffering=1) as log:
        for step in range(1, config["steps"] + 1):
            tick = time.perf_counter()
            visit = (step - 1) // len(scenes)
            index = order[(step - 1) % len(scenes)]
            scene = scenes[index]
            slot = visit % len(scene.record["roles"]["primary_query"])
            optimizer.zero_grad(set_to_none=True)
            # Independently built states share only the trainable slow carrier parameters.
            count = extra_count(config["base_extra"], extra_generator)
            views = {
                role: with_extra_views(scene.context(role), streams[index].stream(role), count)
                if count
                else scene.context(role)
                for role in ("A", "B")
            }
            states = {
                role: build_state(carrier, views[role], scene.bounds[role], f"{seed}:{step}:{role}")
                for role in ("A", "B")
            }
            target = scene.target(slot)  # offline targets only after state construction
            ray_count = int(np.prod(manifest["image_size"]))
            indices, far_counts = {}, {}
            for role in ("A", "B"):
                far = far_pixels(views[role], target, manifest["image_size"])
                indices[role] = far_focus_indices(far, ray_count, config["batch_rays"], generator)
                far_counts[role] = int(far.numel())
            stream.update(json.dumps([scene.record["scene_id"], slot]).encode())
            for role in ("A", "B"):
                stream.update(indices[role].numpy().astype("<i8").tobytes())
            terms = {}
            total = 0
            for role, (state, anchor) in states.items():
                cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                loss, part = training_loss(
                    state, cameras, target.rgb, target.depth, indices[role].to(device), variant
                )
                total = total + loss * 0.5
                terms[role] = {
                    k: float(v.detach()) if isinstance(v, torch.Tensor) else v
                    for k, v in part.items()
                }
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite training objective")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                carrier.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            row = {
                "step": step,
                "scene_id": scene.record["scene_id"],
                "slot": slot,
                "extra_views": count,
                "far_pixels": far_counts,
                "loss": float(total.detach()),
                "gradient_norm": float(norm),
                "seconds": time.perf_counter() - tick,
                "terms": terms,
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 100 == 0 or step == 1:
                print(variant, seed, step, config["steps"], row["loss"], flush=True)
            if step in config["checkpoint_steps"]:
                path = destination / f"step_{step:06d}.pt"
                digest = save_checkpoint(
                    path,
                    carrier,
                    variant=variant,
                    seed=seed,
                    step=step,
                    lock_sha256=sha(root / "preregistration.json"),
                )
                # Evaluator receives only DEV and never FRESH_QUALIFICATION.
                from mcss.mechanism_pilot.rgbd_long_train_evaluation import evaluate_model

                carrier.eval()
                result = evaluate_model(
                    carrier,
                    manifest,
                    root,
                    destination / f"dev_{step:06d}",
                    variant=variant,
                    seed=seed,
                    step=step,
                    device=device,
                    diagnostics=False,
                )
                candidate = {
                    **result["selection"],
                    "partition": "DEV",
                    "step": step,
                    "checkpoint": str(path),
                    "checkpoint_sha256": digest,
                }
                candidates.append(candidate)
                write_json(destination / "dev_curve.json", candidates)
                torch.save(
                    {
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "ray_generator": generator.get_state(),
                        "stream_sha256": stream.hexdigest(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else [],
                    },
                    destination / f"optimizer_{step:06d}.pt",
                )
                carrier.train()
    selection = select_checkpoint(candidates)
    write_json(destination / "selection.json", selection)
    summary = {
        "status": "COMPLETE",
        "variant": variant,
        "seed": seed,
        "steps": config["steps"],
        "initial_state_dict_hash": initial_hash,
        "initial_parameter_hash": initial_parameters,
        "token_count": carrier.config.token_count,
        "depth_log_sigma": (
            float(carrier.depth_log_sigma.detach()) if hasattr(carrier, "depth_log_sigma") else None
        ),
        "data_ray_stream_sha256": stream.hexdigest(),
        "ray_sampling": {
            "rule": "V26 per-role far focus, plan amendment 1 (without replacement)",
            "far_fraction": FAR_FRACTION,
        },
        "selection": selection,
        "wall_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
        if str(device).startswith("cuda")
        else None,
        "parameter_count": sum(p.numel() for p in carrier.parameters()),
        "preregistration_sha256": sha(root / "preregistration.json"),
        "fresh_opened": False,
        "writer_updates": False,
        "dynamic_ttt_run": False,
    }
    verify_lock(root)
    write_json(destination / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.root, args.variant, args.seed, args.device)
