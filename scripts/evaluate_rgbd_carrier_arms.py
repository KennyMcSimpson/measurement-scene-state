#!/usr/bin/env python3
"""Sealed static EVAL of several frozen V17-style carrier arms on new cohorts (no training).

The frozen V17 EVAL stage, generalized from one root's checkpoint labels to arms: each arm is a
finished V17-style root (G1 of V21, P of V31, U of V32, ...) whose step-<k> checkpoints of every
seed are loaded through the arm's own carrier module, hash-checked against the arm's DEV curve
and lock. The evaluation root holds evaluation_contract.json (arms, seeds, cohorts with manifest
hashes, the NEAR radius, the fill constant and the abstention threshold).

1. seal --cohort NAME: for each scene and role, the V11 context-only RGB-D loader gives the
   context and its CONTEXT_DEPTH bounds; every arm and seed builds its state and renders depth
   and opacity at every query camera of the cohort's query role; the context-depth reprojection
   (REPROJ_NN, hit, NEAR mask) and INPAINT-V1's harmonic filling come from the same context.
   Every prediction file is hashed; no query depth is read.
2. score: only after every cohort's seal exists; AbsRel on ALL, NEAR and FAR for CONSTANT,
   REPROJ_NN, REPROJ_HARMONIC and, per arm and seed, the carrier, HFILL8 and AHFILL8.
3. analyze: per cohort and pooled over all cohorts, V11's aggregation, scene bootstrap and V2
   gate for FAR(reference) - FAR(arm), and the INPAINT-V1 label of AbsRel(REPROJ_HARMONIC) -
   AbsRel(AHFILL8 of the arm) and of REPROJ_HARMONIC - arm on FAR.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.geometry_carrier_experiment import read
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

SCRIPTS = Path(__file__).resolve().parent
REGIONS = ("ALL", "NEAR", "FAR")


def frozen_v17_evaluation():
    """The frozen V17 EVAL script: its V11 host (loader, aggregation, gate) and INPAINT module."""
    path = SCRIPTS / "evaluate_rgbd_long_train_eval.py"
    spec = importlib.util.spec_from_file_location("arms_frozen_v17_eval", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def contract(root):
    """The evaluation contract, every arm lock and cohort manifest hash-checked."""
    config = read(Path(root) / "evaluation_contract.json")
    for name, arm in config["arms"].items():
        if sha(Path(arm["root"]) / "preregistration.json") != arm["preregistration_sha256"]:
            raise PermissionError(f"Arm {name} lock changed")
    cohorts = {}
    for name, entry in config["cohorts"].items():
        path = Path(entry["manifest"])
        if sha(path) != entry["manifest_sha256"]:
            raise PermissionError(f"{name} manifest changed")
        manifest = read(path)
        if any(r["split"] != entry["split"] for r in manifest["scenes"]):
            raise PermissionError(f"{name} records only")
        if set(manifest.get("protected_scene_ids", [])) & {
            r["scene_id"] for r in manifest["scenes"]
        }:
            raise PermissionError(f"{name} contains a protected scene")
        cohorts[name] = (entry, manifest, path.parent)
    return config, cohorts


def arm_carriers(config):
    """{(arm, seed): model} at the registered step, hash-checked against each arm's DEV curve."""
    models = {}
    for name, arm in config["arms"].items():
        root, step = Path(arm["root"]), int(arm["step"])
        loader = importlib.import_module(arm["module"]).load_checkpoint
        lock = sha(root / "preregistration.json")
        for seed in config["seeds"]:
            curve = read(root / "checkpoints" / f"L_{seed}" / "dev_curve.json")
            item = next(c for c in curve if int(c["step"]) == step)
            path = Path(item["checkpoint"])
            if sha(path) != item["checkpoint_sha256"]:
                raise PermissionError(f"{name} seed {seed} checkpoint changed")
            model, payload = loader(path, "cpu")
            if payload["lock_sha256"] != lock or payload["step"] != step:
                raise PermissionError(f"{name} seed {seed} checkpoint not from the arm's training")
            models[name, seed] = model.eval()
    return models


def seal(root, cohort):
    root = Path(root).resolve()
    config, cohorts = contract(root)
    entry, manifest, base = cohorts[cohort]
    out = root / "raw" / "eval" / cohort
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("Predictions are sealed once")
    frozen = frozen_v17_evaluation()
    v11, inp = frozen.host(), frozen.inpainting()
    geo = v11.geometry()
    if config["near_px"] != inp.NEAR_PX:
        raise PermissionError("NEAR definition must be V11's")
    models = arm_carriers(config)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    constant = config["reprojection_fill_constant_m"]
    access, entries, states = [], {}, {}
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        for role, context in v11.context_of(record, manifest, base, access).items():
            anchor = context[0].camera.c2w
            bounds = context_depth_bounds(context, anchor)
            points = geo.context_points(context)
            built = {}
            with torch.no_grad():
                for (arm, seed), carrier in models.items():
                    episode = f"{cohort.lower()}:{sid}:{role}"
                    state, state_anchor = frozen.build_state(carrier, context, bounds, episode)
                    built[arm, seed] = (state, state_anchor)
                    states[f"{cohort}:{arm}:{seed}:{sid}/{role}"] = hash_scene_state(state)
            for qid in record["roles"][entry["role"]]:
                access.append(
                    {
                        "cohort": cohort,
                        "scene_id": sid,
                        "frame_id": qid,
                        "purpose": "QUERY_CAMERA_ONLY",
                    }
                )
                camera = geo.query_camera(record, qid, manifest["image_size"])
                depth, hit = geo.reproject(points, camera)
                reproj = geo.fill_nearest(depth, hit, constant) if hit.any() else None
                if reproj is None:
                    reproj = np.full(camera.image_size, constant, dtype=np.float64)
                arrays = {
                    "REPROJ_NN": reproj,
                    "hit": hit,
                    "near": inp.near_mask(hit),
                    "REPROJ_HARMONIC": inp.harmonic_fill(
                        depth, hit, inp.ray_cosine(camera), constant
                    ),
                }
                batched = Cameras(
                    camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
                )
                with torch.no_grad():
                    for (arm, seed), (state, state_anchor) in built.items():
                        local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                        pred = renderer(state, local, ("depth", "visibility"))
                        arrays[f"{arm}_{seed}"] = pred["depth"][0, 0, 0].numpy().astype(np.float64)
                        arrays[f"OPACITY_{arm}_{seed}"] = pred["visibility"][0, 0, 0].numpy()
                path = out / "predictions" / f"{sid}_{role}_{qid}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                entries[f"{cohort}/{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
    if any(
        "depth" in e.get("channels", []) and e.get("purpose") != "CONTEXT_ONLY_RGBD" for e in access
    ):
        raise AssertionError("Only context depth may be read before the seal")
    write_json(out / "state_hashes.json", states)
    write_json(out / "GT_access_before_seal.json", access)
    write_json(
        out / "prediction_seal.json",
        {
            "cohort": cohort,
            "arms": sorted(config["arms"]),
            "seeds": config["seeds"],
            "predictions": entries,
            "state_hashes_sha256": sha(out / "state_hashes.json"),
            "query_depth_read": False,
            "near_px": config["near_px"],
            "reprojection_fill_constant_m": constant,
        },
    )
    print("ARMS_PREDICTIONS_SEALED", cohort, len(entries), flush=True)


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def score(root):
    root = Path(root).resolve()
    config, cohorts = contract(root)
    out = root / "raw" / "eval"
    seals = {
        name: read(out / name / "prediction_seal.json") for name in cohorts
    }  # all sealed first
    threshold = config["abstain_opacity_threshold"]
    rows, access = [], []
    for name, (entry, manifest, _) in cohorts.items():
        seal_file = seals[name]
        constant = seal_file["reprojection_fill_constant_m"]
        for record in manifest["scenes"]:
            sid = record["scene_id"]
            frames = {f["frame_id"]: f for f in record["frames"]}
            for role in ("A", "B"):
                for qid in record["roles"][entry["role"]]:
                    item = seal_file["predictions"][f"{name}/{sid}/{role}/{qid}"]
                    if sha(Path(item["path"])) != item["sha256"]:
                        raise PermissionError("A sealed prediction changed")
                    arrays = dict(np.load(item["path"]))
                    access.append(
                        {
                            "cohort": name,
                            "scene_id": sid,
                            "frame_id": qid,
                            "purpose": "POST_SEAL_QUERY_DEPTH",
                        }
                    )
                    gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                    valid = np.isfinite(gt) & (gt > 0)
                    near = arrays["near"].astype(bool)
                    masks = {"ALL": valid, "NEAR": valid & near, "FAR": valid & ~near}
                    harmonic = arrays["REPROJ_HARMONIC"]
                    predictions = {
                        ("CONSTANT", 0): np.full(gt.shape, constant, dtype=np.float64),
                        ("REPROJ_NN", 0): arrays["REPROJ_NN"],
                        ("REPROJ_HARMONIC", 0): harmonic,
                    }
                    for arm in seal_file["arms"]:
                        for seed in config["seeds"]:
                            carrier = arrays[f"{arm}_{seed}"]
                            low = arrays[f"OPACITY_{arm}_{seed}"].astype(np.float64) < threshold
                            predictions[arm, seed] = carrier
                            predictions[f"HFILL8_{arm}", seed] = np.where(near, harmonic, carrier)
                            predictions[f"AHFILL8_{arm}", seed] = np.where(
                                near | low, harmonic, carrier
                            )
                    common = {"cohort": name, "scene_id": sid, "role": role, "query_id": qid}
                    for (method, seed), value in predictions.items():
                        row = {**common, "method": method, "seed": seed}
                        for region, mask in masks.items():
                            row[f"absrel_{region}"] = absrel(value, gt, mask)
                        row["depth_absrel"] = depth_metrics(value, gt)["depth_absrel"]
                        row["far_fraction"] = float(masks["FAR"].sum() / max(valid.sum(), 1))
                        rows.append(row)
    write_json(out / "query_rows.json", rows)
    write_json(out / "GT_access_after_seal.json", access)
    print("ARMS_SCORED", len(rows), flush=True)


def block(rows, arms, reference):
    """Absolute errors, arm-vs-reference FAR gains and harmonic contrasts over one scene set."""
    frozen = frozen_v17_evaluation()
    v11, label = frozen.host(), frozen.inpainting().geometry().label

    def values(method, region):
        return v11.scene_values(rows, method, f"absrel_{region}")

    def contrast(a, ra, b, rb):
        stats = v11.contrast(values(a, ra), values(b, rb))
        out = {"gain": stats, "definition": f"AbsRel {ra}({a}) - AbsRel {rb}({b})"}
        if stats is not None:
            out["label"] = label(stats)
            out["gate"], out["gate_checks"] = v11.gate(stats)
        return out

    methods = sorted({r["method"] for r in rows})
    contrasts = {}
    for arm in arms:
        if arm != reference:
            contrasts[f"{reference}_MINUS_{arm}_FAR"] = contrast(reference, "FAR", arm, "FAR")
            contrasts[f"{reference}_MINUS_{arm}_ALL"] = contrast(reference, "ALL", arm, "ALL")
            contrasts[f"AHFILL8_{reference}_MINUS_AHFILL8_{arm}"] = contrast(
                f"AHFILL8_{reference}", "ALL", f"AHFILL8_{arm}", "ALL"
            )
        contrasts[f"HARMONIC_MINUS_AHFILL8_{arm}"] = contrast(
            "REPROJ_HARMONIC", "ALL", f"AHFILL8_{arm}", "ALL"
        )
        contrasts[f"HARMONIC_MINUS_{arm}_FAR"] = contrast("REPROJ_HARMONIC", "FAR", arm, "FAR")
    return {
        "absolute_absrel": {
            m: {region: v11.summary(values(m, region)) for region in REGIONS} for m in methods
        },
        "contrasts": contrasts,
        "n_scenes": len({r["scene_id"] for r in rows}),
    }


def analyze(root):
    root = Path(root).resolve()
    config, cohorts = contract(root)
    path = root / "raw" / "eval" / "query_rows.json"
    rows = read(path)
    arms, reference = sorted(config["arms"]), config["reference_arm"]
    result = {
        "pooled": block(rows, arms, reference),
        "cohorts": {
            name: block([r for r in rows if r["cohort"] == name], arms, reference)
            for name in cohorts
        },
        "input_sha256": {str(path): sha(path)},
        "source_sha256": {
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        },
    }
    write_json(root / "arms_results.json", result)
    print("ARMS_ANALYZED", json.dumps(sorted(result["pooled"]["contrasts"])), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("seal", "score", "analyze"), required=True)
    parser.add_argument("--cohort")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "seal":
        seal(args.root, args.cohort)
    elif args.stage == "score":
        score(args.root)
    else:
        analyze(args.root)
