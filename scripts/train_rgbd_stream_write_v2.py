#!/usr/bin/env python3
"""Core B V2: train the write rule of one seed on TRAIN72 over the frozen V9 carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from mcss.dynamic.types import hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_data, verify_lock
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.mechanism_pilot.rgbd_stream_write_v2 import (
    StreamRGBDLoader,
    load_frozen_carrier,
    make_write_rule,
    save_write_rule,
    training_loss,
    unroll,
)
from mcss.mechanism_pilot.small_training import AccessLog, sha, write_json


def train(root, seed, device):
    root = Path(root).resolve()
    manifest, config = verify_lock(root)
    verify_data(root)
    torch.set_num_threads(config["num_threads"])
    if seed not in config["seeds"]:
        raise PermissionError("Unregistered seed")
    lock = read(root / "carrier_lock.json")["carriers"][str(seed)]
    if sha(Path(lock["path"])) != lock["sha256"]:
        raise PermissionError("Sealed V9 carrier checkpoint changed")
    destination = root / "checkpoints" / f"write_{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    access = AccessLog(destination / "access.jsonl")
    carrier, _ = load_frozen_carrier(lock["path"], device)
    carrier_hash = hash_value(carrier.state_dict())
    rule = make_write_rule(carrier, seed)
    lock_sha = sha(root / "preregistration.json")
    initial_sha = save_write_rule(
        destination / "rule_step000000.pt",
        rule,
        seed=seed,
        step=0,
        lock_sha256=lock_sha,
        carrier_sha256=lock["sha256"],
    )
    roles = read(root / "stream_roles.json")
    train_ids = sorted({r["scene_id"] for r in manifest["scenes"] if r["split"] == "TRAIN"})
    holdout = set(manifest.get("protected_scene_ids", [])) | {
        r["scene_id"] for r in manifest["scenes"] if r["split"] != "TRAIN"
    }
    records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == "TRAIN"}
    scenes, streams = {}, {}
    for sid, record in records.items():
        scenes[sid] = DepthBoundsSceneData(record, manifest, root, device, access)
        streams[sid] = StreamRGBDLoader(
            record,
            manifest["image_size"],
            root,
            device,
            access,
            allowed_scene_ids=train_ids,
            holdout_scene_ids=holdout,
        )
    episodes = [tuple(k.split("/")) for k in roles["train_eligible_episodes"]]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(episodes), generator=generator).tolist()
    optimizer = torch.optim.Adam(rule.parameters(), lr=config["learning_rate"])
    stream_hash = hashlib.sha256()
    started = time.perf_counter()
    with (destination / "training.jsonl").open("x", buffering=1) as log:
        for step in range(1, config["steps"] + 1):
            tick = time.perf_counter()
            visit = (step - 1) // len(episodes)
            sid, role = episodes[order[(step - 1) % len(episodes)]]
            scene = scenes[sid]
            slot = visit % len(scene.record["roles"]["primary_query"])
            state, anchor, fast, _ = unroll(
                carrier,
                rule,
                scene.context(role),
                streams[sid].stream(role),
                scene.bounds[role],
                "ALL",
                f"{seed}:{step}:{sid}:{role}",
            )
            target = scene.target(slot)  # offline TRAIN label, after the stream state exists
            cameras = transform_cameras(target.cameras, torch.linalg.inv(anchor))
            ray_count = manifest["image_size"][0] * manifest["image_size"][1]
            indices = torch.randint(ray_count, (config["batch_rays"],), generator=generator)
            stream_hash.update(json.dumps([sid, role, slot]).encode())
            stream_hash.update(indices.numpy().astype("<i8").tobytes())
            loss, terms = training_loss(
                state, cameras, target.rgb, target.depth, indices.to(device)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite write-rule objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                rule.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            row = {
                "step": step,
                "scene_id": sid,
                "role": role,
                "slot": slot,
                "loss": float(loss.detach()),
                "loss_depth": float(terms["loss_depth"].detach()),
                "gradient_norm": float(norm),
                "fast_norm": float((fast.delta_fuse.norm() + fast.delta_complete.norm()).detach()),
                "seconds": time.perf_counter() - tick,
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 250 == 0 or step == 1:
                print(seed, step, config["steps"], round(row["loss_depth"], 4), flush=True)
    if hash_value(carrier.state_dict()) != carrier_hash:
        raise AssertionError("The frozen slow carrier changed during write-rule training")
    final_sha = save_write_rule(
        destination / "rule_final.pt",
        rule,
        seed=seed,
        step=config["steps"],
        lock_sha256=lock_sha,
        carrier_sha256=lock["sha256"],
    )
    summary = {
        "status": "COMPLETE",
        "seed": seed,
        "steps": config["steps"],
        "episodes": len(episodes),
        "initial_rule_sha256": initial_sha,
        "final_rule_sha256": final_sha,
        "carrier_checkpoint_sha256": lock["sha256"],
        "carrier_state_hash_unchanged": True,
        "data_stream_sha256": stream_hash.hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "preregistration_sha256": lock_sha,
        "dev_forward_during_training": False,
    }
    write_json(destination / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.root, args.seed, args.device)
