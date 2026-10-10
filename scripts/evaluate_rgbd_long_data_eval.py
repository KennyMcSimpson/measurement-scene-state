#!/usr/bin/env python3
"""V18 EVAL stage: every prediction is sealed to disk before any query depth is read.

Derived from the V14 EVAL stage. Checkpoints are labelled by step: T<step> is the L checkpoint
at that step and SEL the one the frozen DEV rule selected over all 18 checkpoints.

1. seal --cohort NAME: for each scene and role, the context-only RGB-D loader gives the context
   and its CONTEXT_DEPTH bounds; every registered checkpoint builds states and renders depth and
   opacity at every query camera (primary cohorts: the curve steps and SEL; EVAL-FAR and
   FRESH-V2: T6000, T24000 and SEL); the context-depth reprojection (REPROJ_NN, hit, NEAR mask)
   and INPAINT-V1's harmonic filling come from the same context. Every prediction file is hashed.
2. score: query depth is read only after every cohort's seal exists; AbsRel on ALL, NEAR and FAR.
3. analyze: per-scene values over the pooled EVAL-V3 + EVAL-V4 scenes and the preregistered
   contrasts LENGTH_GAIN and HARMONIC_MINUS_AHFILL8_T24000 (primary) plus descriptive ones; every
   cohort is also reported on its own, with the FAR/NEAR/ALL curve over training steps.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from mcss.dynamic.types import hash_scene_state
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.mechanism_pilot.rgbd_long_data_carrier import build_state, load_checkpoint
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

REGIONS = ("ALL", "NEAR", "FAR")
PRIMARY_COHORTS = ("EVAL_V3", "EVAL_V4")
SCRIPTS = Path(__file__).resolve().parent
CONTRASTS = {
    "LENGTH_GAIN": ("T6000", "FAR", "T24000", "FAR"),
    "HARMONIC_MINUS_AHFILL8_T24000": ("REPROJ_HARMONIC", "ALL", "AHFILL8_T24000", "ALL"),
    "HARMONIC_MINUS_HFILL8_T24000": ("REPROJ_HARMONIC", "ALL", "HFILL8_T24000", "ALL"),
    "HARMONIC_MINUS_AHFILL8_T6000": ("REPROJ_HARMONIC", "ALL", "AHFILL8_T6000", "ALL"),
    "HARMONIC_MINUS_HFILL8_T6000": ("REPROJ_HARMONIC", "ALL", "HFILL8_T6000", "ALL"),
    "HARMONIC_MINUS_T24000_FAR": ("REPROJ_HARMONIC", "FAR", "T24000", "FAR"),
    "HARMONIC_MINUS_T6000_FAR": ("REPROJ_HARMONIC", "FAR", "T6000", "FAR"),
    "AHFILL8_LENGTH_GAIN": ("AHFILL8_T6000", "ALL", "AHFILL8_T24000", "ALL"),
    "HFILL8_LENGTH_GAIN": ("HFILL8_T6000", "ALL", "HFILL8_T24000", "ALL"),
    "T6000_MINUS_T24000_ALL": ("T6000", "ALL", "T24000", "ALL"),
    "T6000_MINUS_T24000_NEAR": ("T6000", "NEAR", "T24000", "NEAR"),
    "T6000_MINUS_SEL_FAR": ("T6000", "FAR", "SEL", "FAR"),
    "HARMONIC_MINUS_AHFILL8_SEL": ("REPROJ_HARMONIC", "ALL", "AHFILL8_SEL", "ALL"),
    "CONSTANT_MINUS_T24000": ("CONSTANT", "ALL", "T24000", "ALL"),
    "NN_MINUS_T24000": ("REPROJ_NN", "ALL", "T24000", "ALL"),
}
DEFINITIONS = {
    "LENGTH_GAIN": "FAR AbsRel(L at step 6000) - FAR AbsRel(L at step 24000), "
    "pooled EVAL-V3 + EVAL-V4",
    "HARMONIC_MINUS_AHFILL8_T24000": "AbsRel(REPROJ_HARMONIC) - AbsRel(AHFILL8 of L at step 24000)",
    "FAR": "valid pixels farther than 8 px from any context-depth reprojection hit",
    "HFILL8": "REPROJ_HARMONIC within 8 px of a hit, the carrier elsewhere",
    "AHFILL8": "HFILL8 whose FAR pixels of rendered opacity < 0.5 also take REPROJ_HARMONIC",
    "T<step>": "the L checkpoint at that step; T6000 equals V14 C1's last step (built-in control)",
    "SEL": "the checkpoint the frozen V2 DEV rule selected over all 18 checkpoints",
}


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@functools.cache
def host():
    """V11's frozen EVAL-V3 module: context loader, aggregation, bootstrap and V2 gate."""
    return module("v18_host_v11_eval", "evaluate_rgbd_completion_eval_v3.py")


@functools.cache
def inpainting():
    """INPAINT-V1's frozen module: ray cosine, harmonic filling, NEAR mask, label rule."""
    return module("v18_inpaint", "run_rgbd_inpaint_baselines.py")


def cohorts(root):
    """The frozen contract and {cohort: (entry, manifest, base)}, manifests hash-checked."""
    config = read(root / "training_contract.json")
    frozen = {}
    for name, entry in config["eval"]["cohorts"].items():
        path = Path(entry["manifest"])
        if sha(path) != entry["manifest_sha256"]:
            raise PermissionError(f"{name} manifest changed after the freeze")
        manifest = read(path)
        if any(r["split"] != entry["split"] for r in manifest["scenes"]):
            raise PermissionError(f"{name} records only")
        frozen[name] = (entry, manifest, path.parent)
    return config, frozen


def labels_for(name, config):
    steps = (
        config["eval"]["curve_steps_primary"]
        if name in PRIMARY_COHORTS
        else config["eval"]["steps_descriptive_cohorts"]
    )
    return [f"T{step}" for step in steps] + ["SEL"]


def checkpoint_path(root, seed, label):
    """The hash-checked checkpoint behind a label, from the run's own DEV curve and selection."""
    run = root / "checkpoints" / f"L_{seed}"
    curve = {c["step"]: c for c in read(run / "dev_curve.json")}
    if label == "SEL":
        selection = read(run / "selection.json")
        step = selection["selected_step"]
        if selection["checkpoint_sha256"] != curve[step]["checkpoint_sha256"]:
            raise PermissionError(f"Selection and DEV curve disagree: seed {seed}")
    else:
        step = int(label[1:])
    item = curve[step]
    if sha(Path(item["checkpoint"])) != item["checkpoint_sha256"]:
        raise PermissionError(f"Checkpoint changed: seed {seed} step {step}")
    return Path(item["checkpoint"]), step


def carriers(root, config, labels):
    """{(label, seed): model} for the registered checkpoints (hash- and lock-checked)."""
    models, steps = {}, {}
    lock = sha(root / "preregistration.json")
    for seed in config["seeds"]:
        for label in labels:
            path, step = checkpoint_path(root, seed, label)
            model, payload = load_checkpoint(path, "cpu")
            if payload["lock_sha256"] != lock or payload["step"] != step:
                raise PermissionError("Checkpoint not from the frozen V18 training")
            models[label, seed] = model.eval()
            steps[f"{label}_{seed}"] = step
    return models, steps


@torch.no_grad()
def seal(root, cohort):
    root = Path(root).resolve()
    verify_lock(root)
    if read(root / "audit/v14_prefix_control.json")["status"] != "PASS":
        raise PermissionError("The built-in V14 C1 control must PASS before any EVAL prediction")
    config, frozen = cohorts(root)
    entry, manifest, base = frozen[cohort]
    out = root / "raw" / "eval" / cohort
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("EVAL predictions are sealed once")
    v11, inp = host(), inpainting()
    geo = v11.geometry()
    labels = labels_for(cohort, config)
    models, steps = carriers(root, config, labels)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    constant = config["eval"]["reprojection_fill_constant_m"]
    if config["eval"]["near_px"] != inp.NEAR_PX:
        raise PermissionError("NEAR definition must be V11's")
    access, entries, states = [], {}, {}
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        for role, context in v11.context_of(record, manifest, base, access).items():
            anchor = context[0].camera.c2w
            bounds = context_depth_bounds(context, anchor)
            points = geo.context_points(context)
            built = {}
            for (label, seed), carrier in models.items():
                episode = f"{cohort.lower()}:{sid}:{role}"
                state, state_anchor = build_state(carrier, context, bounds, episode)
                built[label, seed] = (state, state_anchor)
                states[f"{cohort}:{label}:{seed}:{sid}/{role}"] = hash_scene_state(state)
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
                cosine = inp.ray_cosine(camera)
                arrays = {
                    "REPROJ_NN": reproj,
                    "hit": hit,
                    "near": inp.near_mask(hit),
                    "REPROJ_HARMONIC": inp.harmonic_fill(depth, hit, cosine, constant),
                }
                batched = Cameras(
                    camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
                )
                for (label, seed), (state, state_anchor) in built.items():
                    local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                    pred = renderer(state, local, ("depth", "visibility"))
                    arrays[f"{label}_{seed}"] = pred["depth"][0, 0, 0].numpy().astype(np.float64)
                    arrays[f"OPACITY_{label}_{seed}"] = pred["visibility"][0, 0, 0].numpy()
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
            "labels": labels,
            "checkpoint_steps": steps,
            "predictions": entries,
            "state_hashes_sha256": sha(out / "state_hashes.json"),
            "query_depth_read": False,
            "near_px": config["eval"]["near_px"],
            "reprojection_fill_constant_m": constant,
        },
    )
    print("EVAL_PREDICTIONS_SEALED", cohort, len(entries), flush=True)


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def score(root):
    root = Path(root).resolve()
    config, frozen = cohorts(root)
    out = root / "raw" / "eval"
    seals = {name: read(out / name / "prediction_seal.json") for name in frozen}  # all sealed first
    threshold = config["eval"]["abstain_opacity_threshold"]
    rows, access = [], []
    for name, (entry, manifest, _) in frozen.items():
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
                    for label in seal_file["labels"]:
                        for seed in config["seeds"]:
                            carrier = arrays[f"{label}_{seed}"]
                            low = arrays[f"OPACITY_{label}_{seed}"].astype(np.float64) < threshold
                            predictions[label, seed] = carrier
                            predictions[f"HFILL8_{label}", seed] = np.where(near, harmonic, carrier)
                            predictions[f"AHFILL8_{label}", seed] = np.where(
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
    print("EVAL_SCORED", len(rows), flush=True)


def block(rows):
    """Absolute errors and every contrast over one set of scenes (V11 aggregation, labels)."""
    v11, label = host(), inpainting().geometry().label

    def values(method, region):
        return v11.scene_values(rows, method, f"absrel_{region}")

    methods = sorted({r["method"] for r in rows})
    contrasts = {}
    for name, (a, ra, b, rb) in CONTRASTS.items():
        if a not in methods or b not in methods:
            continue
        stats = v11.contrast(values(a, ra), values(b, rb))
        contrasts[name] = {"gain": stats, "definition": f"AbsRel {ra}({a}) - AbsRel {rb}({b})"}
        if stats is not None:
            contrasts[name]["label"] = label(stats)
            contrasts[name]["gate"], contrasts[name]["gate_checks"] = v11.gate(stats)
    curve = {}
    for m in methods:
        if m.startswith("T") and m[1:].isdigit():
            curve[int(m[1:])] = {region: v11.summary(values(m, region)) for region in REGIONS}
    return {
        "absolute_absrel": {
            m: {region: v11.summary(values(m, region)) for region in REGIONS} for m in methods
        },
        "contrasts": contrasts,
        "curve": {str(k): curve[k] for k in sorted(curve)},
        "n_scenes": len({r["scene_id"] for r in rows}),
        "far_pixel_fraction": float(
            np.mean([r["far_fraction"] for r in rows if r["method"] == "REPROJ_NN"])
        ),
    }


def analyze_rows(rows):
    primary = block([r for r in rows if r["cohort"] in PRIMARY_COHORTS])
    contrasts = primary["contrasts"]
    statuses = {
        "LENGTH_STATUS": contrasts.get("LENGTH_GAIN", {}).get("gate", "UNDEFINED"),
        "AHFILL8_LABEL": contrasts.get("HARMONIC_MINUS_AHFILL8_T24000", {}).get(
            "label", "UNDEFINED"
        ),
    }
    first = statuses["LENGTH_STATUS"] == "SUPPORTED"
    second = statuses["AHFILL8_LABEL"] == "ABOVE"
    names = sorted({r["cohort"] for r in rows})
    return {
        "primary": primary,
        "cohorts": {name: block([r for r in rows if r["cohort"] == name]) for name in names},
        "statuses": statuses,
        "INTERPRETATION_BRANCH": "A" if first and second else "B" if first or second else "C",
        "definitions": DEFINITIONS,
    }


def analyze(root, output=None):
    root = Path(root).resolve()
    rows_path = root / "raw" / "eval" / "query_rows.json"
    result = analyze_rows(read(rows_path))
    result["control"] = read(root / "audit/v14_prefix_control.json")["status"]
    result["input_sha256"] = {str(rows_path): sha(rows_path)}
    result["source_sha256"] = {
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    target = Path(output) if output else root / "eval_results.json"
    write_json(target, result)
    print(
        "EVAL_BRANCH", result["INTERPRETATION_BRANCH"], json.dumps(result["statuses"]), flush=True
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("seal", "score", "analyze"), required=True)
    parser.add_argument("--cohort")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "seal":
        seal(args.root, args.cohort)
    elif args.stage == "score":
        score(args.root)
    else:
        analyze(args.root, args.output)
