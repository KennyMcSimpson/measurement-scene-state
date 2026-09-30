#!/usr/bin/env python3
"""V12 carrier-width runs of the V11 C1 recipe; no writer or test cohort access."""

from __future__ import annotations

import argparse
import hashlib
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
from mcss.mechanism_pilot.rgbd_stream_write import StreamRGBDLoader
from mcss.mechanism_pilot.rgbd_width_carrier import (
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
from mcss.mechanism_pilot.small_training import AccessLog, sha, write_json


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
            states = {
                role: build_state(
                    carrier,
                    with_extra_views(scene.context(role), streams[index].stream(role), count)
                    if count
                    else scene.context(role),
                    scene.bounds[role],
                    f"{seed}:{step}:{role}",
                )
                for role in ("A", "B")
            }
            target = scene.target(slot)  # offline targets only after state construction
            ray_count = int(np.prod(manifest["image_size"]))
            indices = torch.randint(ray_count, (config["batch_rays"],), generator=generator)
            stream.update(json.dumps([scene.record["scene_id"], slot]).encode())
            stream.update(indices.numpy().astype("<i8").tobytes())
            terms = {}
            total = 0
            for role, (state, anchor) in states.items():
                cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
                loss, part = training_loss(
                    state, cameras, target.rgb, target.depth, indices.to(device), variant
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
                from mcss.mechanism_pilot.rgbd_width_evaluation import evaluate_model

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
