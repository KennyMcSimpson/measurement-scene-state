#!/usr/bin/env python3
"""V15 EVAL stage: every prediction is sealed to disk before any query depth is read.

1. seal: for each scene and role of EVAL-V3 and EVAL-V4 (primary queries) and EVAL-FAR (far
   queries), the context-only RGB-D loader gives the context and its CONTEXT_DEPTH bounds; V15
   C0/C1's selected checkpoints build states and render depth and opacity at every query camera.
   On the primary cohorts, V12 C1's selected checkpoints (width 64, TRAIN72) and the step-6000
   checkpoints of C0/C1 are rendered as well (descriptive). The context-depth reprojection and
   INPAINT-V1's harmonic filling come from the same context. Every prediction file is hashed. C0
   must reproduce the sealed V14 C1 predictions of every cohort or the stage stops.
2. score: query depth is read only after the seal file exists; the width-32 / TRAIN72 cell is
   V14's sealed C0 (= V11 C1) prediction; AbsRel on ALL, NEAR and FAR.
3. analyze: over the pooled EVAL-V3 + EVAL-V4 scenes the primary WIDTH_AT_SCALE_GAIN and
   HARMONIC_HYBRID_GAIN, the secondary HFILL8_WIDTH_GAIN and the descriptive width x data 2x2;
   every cohort is also reported on its own (EVAL-FAR descriptive).
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
from mcss.mechanism_pilot import rgbd_width_carrier as v12
from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.mechanism_pilot.rgbd_wide_scale_carrier import PRIMARY_PAIR, build_state, load_checkpoint
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

REGIONS = ("ALL", "NEAR", "FAR")
PRIMARY_COHORTS = ("EVAL_V3", "EVAL_V4")
SCRIPTS = Path(__file__).resolve().parent
# Descriptive methods rendered on the primary cohorts only (array prefix -> row method).
EXTRA = {"W64T72": "W64_T72", "C0LAST": "C0_LAST", "C1LAST": "C1_LAST"}
CONTRASTS = {
    "WIDTH_AT_SCALE_GAIN": ("C0", "FAR", "C1", "FAR"),
    "HARMONIC_HYBRID_GAIN": ("REPROJ_HARMONIC", "ALL", "HFILL8_C1", "ALL"),
    "HFILL8_WIDTH_GAIN": ("HFILL8_C0", "ALL", "HFILL8_C1", "ALL"),
    "WIDTH_AT_T72_GAIN": ("W32_T72", "FAR", "W64_T72", "FAR"),
    "DATA_AT_W32_GAIN": ("W32_T72", "FAR", "C0", "FAR"),
    "DATA_AT_W64_GAIN": ("W64_T72", "FAR", "C1", "FAR"),
    "C0_MINUS_C1_ALL": ("C0", "ALL", "C1", "ALL"),
    "C0_MINUS_C1_NEAR": ("C0", "NEAR", "C1", "NEAR"),
    "HARMONIC_MINUS_C1_FAR": ("REPROJ_HARMONIC", "FAR", "C1", "FAR"),
    "HARMONIC_MINUS_HFILL8_C0": ("REPROJ_HARMONIC", "ALL", "HFILL8_C0", "ALL"),
    "HARMONIC_MINUS_HFILL8_W64_T72": ("REPROJ_HARMONIC", "ALL", "HFILL8_W64_T72", "ALL"),
    "AHFILL8_WIDTH_GAIN": ("AHFILL8_C0", "ALL", "AHFILL8_C1", "ALL"),
    "HARMONIC_MINUS_AHFILL8_C1": ("REPROJ_HARMONIC", "ALL", "AHFILL8_C1", "ALL"),
    "LAST_WIDTH_GAIN": ("C0_LAST", "FAR", "C1_LAST", "FAR"),
    "HARMONIC_MINUS_HFILL8_C1_LAST": ("REPROJ_HARMONIC", "ALL", "HFILL8_C1_LAST", "ALL"),
    "HARMONIC_MINUS_HFILL8_C0_LAST": ("REPROJ_HARMONIC", "ALL", "HFILL8_C0_LAST", "ALL"),
    "CONSTANT_MINUS_C1": ("CONSTANT", "ALL", "C1", "ALL"),
    "NN_MINUS_C1": ("REPROJ_NN", "ALL", "C1", "ALL"),
}
DEFINITIONS = {
    "WIDTH_AT_SCALE_GAIN": "FAR AbsRel(C0 width 32) - FAR AbsRel(C1 width 64), both TRAIN72+EXT",
    "HARMONIC_HYBRID_GAIN": "AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of C1), same scenes",
    "HFILL8_WIDTH_GAIN": "AbsRel(HFILL8 of C0) - AbsRel(HFILL8 of C1), same scenes",
    "INTERACTION": "[FAR(32,T72) - FAR(64,T72)] - [FAR(32,T127) - FAR(64,T127)] per scene; < 0 "
    "means width helps more with more data",
    "W32_T72": "V14's sealed C0 (= V11 C1) predictions",
    "W64_T72": "V12 C1 selected checkpoints (width 64, TRAIN72)",
    "LAST": "the step-6000 checkpoints of C0 and C1 (descriptive)",
    "FAR": "valid pixels farther than 8 px from any context-depth reprojection hit",
    "HFILL8": "REPROJ_HARMONIC within 8 px of a hit, the carrier elsewhere",
    "AHFILL8": "HFILL8 whose FAR pixels of rendered opacity < 0.5 also take REPROJ_HARMONIC",
}


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@functools.cache
def host():
    """V11's frozen EVAL-V3 module: context loader, aggregation, bootstrap and V2 gate."""
    return module("v15_host_v11_eval", "evaluate_rgbd_completion_eval_v3.py")


@functools.cache
def inpainting():
    """INPAINT-V1's frozen module: ray cosine, harmonic filling, NEAR mask, label rule."""
    return module("v15_inpaint", "run_rgbd_inpaint_baselines.py")


def cohorts(root):
    """The frozen contract and {cohort: (entry, manifest, base)}, manifests hash-checked."""
    config = read(root / "training_contract.json")
    frozen = {}
    for name, entry in config["eval"]["cohorts"].items():
        path = Path(entry["manifest"])
        if sha(path) != entry["manifest_sha256"]:
            raise PermissionError(f"{name} manifest changed after the freeze")
        manifest = read(path)
        if any(r["split"] != name for r in manifest["scenes"]):
            raise PermissionError(f"{name} records only")
        frozen[name] = (entry, manifest, path.parent)
    return config, frozen


def carriers(root, config):
    """(main, extra): V15 C0/C1 selected checkpoints; V12 C1 selected and C0/C1 step-6000."""
    selected = read(root / "selected_checkpoints.json")
    lock = sha(root / "preregistration.json")
    main, extra = {}, {}
    for variant in PRIMARY_PAIR:
        for seed in config["seeds"]:
            item = selected[variant][str(seed)]
            if sha(Path(item["path"])) != item["sha256"]:
                raise PermissionError("Selected checkpoint changed")
            model, payload = load_checkpoint(item["path"], "cpu")
            if payload["lock_sha256"] != lock:
                raise PermissionError("Checkpoint not from frozen training")
            main[variant, seed] = model.eval()
            last = (
                root
                / "checkpoints"
                / f"{variant}_{seed}"
                / f"step_{config['eval']['last_step']:06d}.pt"
            )
            model, payload = load_checkpoint(last, "cpu")
            if payload["lock_sha256"] != lock or payload["step"] != config["eval"]["last_step"]:
                raise PermissionError("Last-step checkpoint not from frozen training")
            extra[f"{variant}LAST", seed] = model.eval()
    control = config["eval"]["width64_train72"]
    if sha(Path(control["selected"])) != control["selected_sha256"]:
        raise PermissionError("V12 selected checkpoints changed")
    for seed in config["seeds"]:
        item = read(control["selected"])["C1"][str(seed)]
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError("A V12 C1 checkpoint changed")
        extra["W64T72", seed] = v12.load_checkpoint(item["path"], "cpu")[0].eval()
    return main, extra


def consistency(entries, config):
    """V15 C0 against the sealed V14 C1 predictions of every cohort (npz hashes verified)."""
    worst, compared, seals = 0.0, 0, {}
    for key, mine in entries.items():
        name = key.split("/", 1)[0]
        path = config["eval"]["cohorts"][name]["v14_seal"]
        if path not in seals:
            seals[path] = read(path)["predictions"]
        sealed_entry = seals[path][key]
        if sha(Path(sealed_entry["path"])) != sealed_entry["sha256"]:
            raise PermissionError(f"A sealed V14 {name} prediction changed")
        sealed, arrays = dict(np.load(sealed_entry["path"])), dict(np.load(mine["path"]))
        for seed in config["seeds"]:
            worst = max(worst, float(np.max(np.abs(arrays[f"C0_{seed}"] - sealed[f"C1_{seed}"]))))
        compared += 1
    return {"c0_minus_v14_c1_max_abs": worst, "queries_compared": compared}


@torch.no_grad()
def seal(root):
    root = Path(root).resolve()
    verify_lock(root)
    config, frozen = cohorts(root)
    out = root / "raw" / "eval"
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("EVAL predictions are sealed once")
    v11, inp = host(), inpainting()
    geo = v11.geometry()
    main, extra = carriers(root, config)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    constant = config["eval"]["reprojection_fill_constant_m"]
    if config["eval"]["near_px"] != inp.NEAR_PX:
        raise PermissionError("NEAR definition must be V11's")
    access, entries, states = [], {}, {}
    for name, (entry, manifest, base) in frozen.items():
        models = {**main, **extra} if name in PRIMARY_COHORTS else main
        for record in manifest["scenes"]:
            sid = record["scene_id"]
            for role, context in v11.context_of(record, manifest, base, access).items():
                anchor = context[0].camera.c2w
                bounds = context_depth_bounds(context, anchor)
                points = geo.context_points(context)
                built = {}
                for (method, seed), carrier in models.items():
                    episode = f"{name.lower()}:{sid}:{role}"
                    builder = v12.build_state if method == "W64T72" else build_state
                    state, state_anchor = builder(carrier, context, bounds, episode)
                    built[method, seed] = (state, state_anchor)
                    states[f"{name}:{method}:{seed}:{sid}/{role}"] = hash_scene_state(state)
                for qid in record["roles"][entry["role"]]:
                    access.append(
                        {
                            "cohort": name,
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
                    for (method, seed), (state, state_anchor) in built.items():
                        local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                        pred = renderer(state, local, ("depth", "visibility"))
                        arrays[f"{method}_{seed}"] = (
                            pred["depth"][0, 0, 0].numpy().astype(np.float64)
                        )
                        arrays[f"OPACITY_{method}_{seed}"] = pred["visibility"][0, 0, 0].numpy()
                    path = out / "predictions" / name / f"{sid}_{role}_{qid}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **arrays)
                    entries[f"{name}/{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
    if any(
        "depth" in e.get("channels", []) and e.get("purpose") != "CONTEXT_ONLY_RGBD" for e in access
    ):
        raise AssertionError("Only context depth may be read before the seal")
    checked = consistency(entries, config)
    if not checked["c0_minus_v14_c1_max_abs"] <= config["eval"]["c0_reproduction_tolerance"]:
        raise AssertionError(f"V15 C0 does not reproduce the sealed V14 C1 predictions: {checked}")
    write_json(out / "state_hashes.json", states)
    write_json(out / "GT_access_before_seal.json", access)
    write_json(
        out / "prediction_seal.json",
        {
            "predictions": entries,
            "state_hashes_sha256": sha(out / "state_hashes.json"),
            "query_depth_read": False,
            "near_px": config["eval"]["near_px"],
            "reprojection_fill_constant_m": constant,
            "consistency": checked,
        },
    )
    print("EVAL_PREDICTIONS_SEALED", len(entries), json.dumps(checked), flush=True)


def absrel(value, gt, mask):
    return float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def score(root):
    root = Path(root).resolve()
    config, frozen = cohorts(root)
    out = root / "raw" / "eval"
    seal_file = read(out / "prediction_seal.json")
    constant = seal_file["reprojection_fill_constant_m"]
    threshold = config["eval"]["abstain_opacity_threshold"]
    rows, access, v14 = [], [], {}
    for name, (entry, manifest, _) in frozen.items():
        for record in manifest["scenes"]:
            sid = record["scene_id"]
            frames = {f["frame_id"]: f for f in record["frames"]}
            for role in ("A", "B"):
                for qid in record["roles"][entry["role"]]:
                    key = f"{name}/{sid}/{role}/{qid}"
                    item = seal_file["predictions"][key]
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
                    carriers_ = {}
                    for variant in PRIMARY_PAIR:
                        for seed in config["seeds"]:
                            carriers_[variant, seed] = (
                                arrays[f"{variant}_{seed}"],
                                arrays[f"OPACITY_{variant}_{seed}"].astype(np.float64),
                            )
                    if name in PRIMARY_COHORTS:
                        path = entry["v14_seal"]
                        if path not in v14:
                            v14[path] = read(path)["predictions"]
                        theirs = v14[path][key]
                        if sha(Path(theirs["path"])) != theirs["sha256"]:
                            raise PermissionError("A sealed V14 prediction changed")
                        sealed = dict(np.load(theirs["path"]))
                        for seed in config["seeds"]:
                            carriers_["W32_T72", seed] = (sealed[f"C0_{seed}"], None)
                            for prefix, method in EXTRA.items():
                                carriers_[method, seed] = (arrays[f"{prefix}_{seed}"], None)
                    predictions = {
                        ("CONSTANT", 0): np.full(gt.shape, constant, dtype=np.float64),
                        ("REPROJ_NN", 0): arrays["REPROJ_NN"],
                        ("REPROJ_HARMONIC", 0): harmonic,
                    }
                    for (method, seed), (carrier, opacity) in carriers_.items():
                        predictions[method, seed] = carrier
                        predictions[f"HFILL8_{method}", seed] = np.where(near, harmonic, carrier)
                        if opacity is not None:
                            low = opacity < threshold
                            predictions[f"AHFILL8_{method}", seed] = np.where(
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
    """Absolute errors, every contrast and the width x data interaction over one set of scenes."""
    v11, label = host(), inpainting().geometry().label

    def values(method, region):
        return v11.scene_values(rows, method, f"absrel_{region}")

    def entry(stats, definition):
        out = {"gain": stats, "definition": definition}
        if stats is not None:
            out["label"] = label(stats)
            out["gate"], out["gate_checks"] = v11.gate(stats)
        return out

    contrasts = {
        name: entry(
            v11.contrast(values(a, ra), values(b, rb)), f"AbsRel {ra}({a}) - AbsRel {rb}({b})"
        )
        for name, (a, ra, b, rb) in CONTRASTS.items()
    }
    cells = {m: values(m, "FAR") for m in ("W32_T72", "W64_T72", "C0", "C1")}
    common = (
        sorted(set.intersection(*(set(v) for v in cells.values()))) if all(cells.values()) else []
    )
    at72 = {s: cells["W32_T72"][s] - cells["W64_T72"][s] for s in common}
    at127 = {s: cells["C0"][s] - cells["C1"][s] for s in common}
    contrasts["INTERACTION"] = entry(
        v11.contrast(at72, at127) if common else None, DEFINITIONS["INTERACTION"]
    )
    methods = sorted({r["method"] for r in rows})
    return {
        "absolute_absrel": {
            m: {region: v11.summary(values(m, region)) for region in REGIONS} for m in methods
        },
        "contrasts": contrasts,
        "n_scenes": len({r["scene_id"] for r in rows}),
        "far_pixel_fraction": float(
            np.mean([r["far_fraction"] for r in rows if r["method"] == "REPROJ_NN"])
        ),
    }


def analyze_rows(rows):
    primary = block([r for r in rows if r["cohort"] in PRIMARY_COHORTS])
    contrasts = primary["contrasts"]
    statuses = {
        "WIDTH_AT_SCALE_STATUS": contrasts["WIDTH_AT_SCALE_GAIN"].get("gate", "UNDEFINED"),
        "HARMONIC_HYBRID_LABEL": contrasts["HARMONIC_HYBRID_GAIN"].get("label", "UNDEFINED"),
        "HFILL8_WIDTH_STATUS": contrasts["HFILL8_WIDTH_GAIN"].get("gate", "UNDEFINED"),
    }
    first = statuses["WIDTH_AT_SCALE_STATUS"] == "SUPPORTED"
    second = statuses["HARMONIC_HYBRID_LABEL"] == "ABOVE"
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
    result["consistency"] = read(root / "raw" / "eval" / "prediction_seal.json")["consistency"]
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"seal": seal, "score": score}.get(args.stage, lambda root: analyze(root, args.output))(
        args.root
    )
