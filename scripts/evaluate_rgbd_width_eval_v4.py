#!/usr/bin/env python3
"""V12 EVAL-V4 stage: every prediction is sealed to disk before any query depth is read.

1. seal: for each EVAL-V4 scene and role, the context-only RGB-D loader gives the context and
   its CONTEXT_DEPTH bounds; V11 C1's selected checkpoints (the plan's control, V11 loader) and
   V12 C0/C1's selected checkpoints build states and render every primary query camera; the
   context-depth reprojection (REPROJ_NN, hit, NEAR mask) and INPAINT-V1's harmonic and Telea
   fillings come from the same context. Every prediction file is hashed.
2. score: query depth is read only after the seal file exists; AbsRel on ALL, NEAR and FAR.
3. analyze: per-scene values and the preregistered contrasts: WIDTH64_COMPLETION_GAIN (primary),
   HARMONIC_HYBRID_GAIN and HFILL8_WIDTH_GAIN (secondary).
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
from mcss.mechanism_pilot.rgbd_width_carrier import (
    PRIMARY_PAIR,
    build_state,
    load_checkpoint,
    load_v11_checkpoint,
)
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

REGIONS = ("ALL", "NEAR", "FAR")
CONTROL = "V11C1"
SCRIPTS = Path(__file__).resolve().parent
REPLICATION = Path("outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4")


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@functools.cache
def host():
    """V11's frozen EVAL-V3 module: context loader, aggregation, bootstrap and V2 gate."""
    return module("v12_host_v11_eval", "evaluate_rgbd_completion_eval_v3.py")


@functools.cache
def inpainting():
    """INPAINT-V1's frozen module: ray cosine, harmonic/Telea filling, NEAR mask, label rule."""
    return module("v12_inpaint", "run_rgbd_inpaint_baselines.py")


def cohort(root):
    """The verified EVAL-V4 manifest frozen by prepare (hash-checked)."""
    config = read(root / "training_contract.json")
    entry = config["eval_v4"]
    path = Path(entry["manifest"])
    if sha(path) != entry["manifest_sha256"]:
        raise PermissionError("EVAL-V4 manifest changed after the freeze")
    manifest = read(path)
    if any(r["split"] != "EVAL_V4" for r in manifest["scenes"]):
        raise PermissionError("EVAL-V4 records only")
    return config, manifest, path.parent


def carriers(root, config):
    """{(method, seed): model}: V11 C1 control first, then V12 C0 and C1 (all hash-checked)."""
    control = config["eval_v4"]["control"]
    if sha(Path(control["selected"])) != control["selected_sha256"]:
        raise PermissionError("V11 selected checkpoints changed")
    models = {}
    v11_selected = read(control["selected"])["C1"]
    for seed in config["seeds"]:
        item = v11_selected[str(seed)]
        if sha(Path(item["path"])) != item["sha256"]:
            raise PermissionError("A V11 C1 checkpoint changed")
        model, _ = load_v11_checkpoint(item["path"], "cpu")
        models[CONTROL, seed] = model.eval()
    selected = read(root / "selected_checkpoints.json")
    for variant in PRIMARY_PAIR:
        for seed in config["seeds"]:
            item = selected[variant][str(seed)]
            if sha(Path(item["path"])) != item["sha256"]:
                raise PermissionError("Selected checkpoint changed")
            model, payload = load_checkpoint(item["path"], "cpu")
            if payload["lock_sha256"] != sha(root / "preregistration.json"):
                raise PermissionError("Checkpoint not from frozen training")
            models[variant, seed] = model.eval()
    return models


@torch.no_grad()
def seal(root):
    root = Path(root).resolve()
    verify_lock(root)
    config, manifest, base = cohort(root)
    out = root / "raw" / "eval_v4"
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("EVAL-V4 predictions are sealed once")
    v11, inp = host(), inpainting()
    geo = v11.geometry()
    models = carriers(root, config)
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    constant = config["eval_v4"]["reprojection_fill_constant_m"]
    if config["eval_v4"]["near_px"] != inp.NEAR_PX:
        raise PermissionError("NEAR definition must be V11's")
    access, entries, states = [], {}, {}
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        for role, context in v11.context_of(record, manifest, base, access).items():
            anchor = context[0].camera.c2w
            bounds = context_depth_bounds(context, anchor)
            points = geo.context_points(context)
            built = {}
            for (method, seed), carrier in models.items():
                state, state_anchor = build_state(carrier, context, bounds, f"eval_v4:{sid}:{role}")
                built[method, seed] = (state, state_anchor)
                states[f"{method}:{seed}:{sid}/{role}"] = hash_scene_state(state)
            for qid in record["roles"]["primary_query"]:
                access.append({"scene_id": sid, "frame_id": qid, "purpose": "QUERY_CAMERA_ONLY"})
                camera = geo.query_camera(record, qid, manifest["image_size"])
                depth, hit = geo.reproject(points, camera)
                reproj = geo.fill_nearest(depth, hit, constant) if hit.any() else None
                if reproj is None:
                    reproj = np.full(camera.image_size, constant, dtype=np.float64)
                cosine = inp.ray_cosine(camera)
                telea, fallback = inp.telea_fill(depth, hit, cosine, constant)
                arrays = {
                    "REPROJ_NN": reproj,
                    "hit": hit,
                    "near": inp.near_mask(hit),
                    "REPROJ_HARMONIC": inp.harmonic_fill(depth, hit, cosine, constant),
                    "REPROJ_TELEA": telea,
                    "telea_fallback": np.array(fallback),
                }
                batched = Cameras(
                    camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
                )
                for (method, seed), (state, state_anchor) in built.items():
                    local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                    pred = renderer(state, local, ("depth",))["depth"][0, 0, 0].numpy()
                    arrays[f"{method}_{seed}"] = pred.astype(np.float64)
                path = out / "predictions" / f"{sid}_{role}_{qid}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                entries[f"{sid}/{role}/{qid}"] = {"path": str(path), "sha256": sha(path)}
    if any(
        "depth" in e.get("channels", []) and e.get("purpose") != "CONTEXT_ONLY_RGBD" for e in access
    ):
        raise AssertionError("Only context depth may be read before the seal")
    write_json(out / "state_hashes.json", states)
    write_json(out / "GT_access_before_seal.json", access)
    write_json(
        out / "prediction_seal.json",
        {
            "predictions": entries,
            "state_hashes_sha256": sha(out / "state_hashes.json"),
            "query_depth_read": False,
            "near_px": config["eval_v4"]["near_px"],
            "reprojection_fill_constant_m": constant,
            "consistency": consistency(entries, config["seeds"]),
        },
    )
    print("EVAL_V4_PREDICTIONS_SEALED", len(entries), flush=True)


def consistency(entries, seeds, replication=REPLICATION):
    """V12 C0 against the V11 C1 control, and the control against the replication's seal."""
    retrain, control, compared = 0.0, None, 0
    other = replication / "raw" / "prediction_seal.json"
    theirs = read(other)["predictions"] if other.exists() else None
    for key, entry in entries.items():
        arrays = dict(np.load(entry["path"]))
        for seed in seeds:
            retrain = max(
                retrain, float(np.max(np.abs(arrays[f"C0_{seed}"] - arrays[f"{CONTROL}_{seed}"])))
            )
        if theirs is not None:
            replicated = dict(np.load(theirs[key]["path"]))
            for seed in seeds:
                diff = float(np.max(np.abs(replicated[f"C1_{seed}"] - arrays[f"{CONTROL}_{seed}"])))
                control = diff if control is None else max(control, diff)
            for name in ("REPROJ_NN", "REPROJ_HARMONIC", "REPROJ_TELEA"):
                diff = float(np.max(np.abs(replicated[name] - arrays[name])))
                control = max(control, diff)
            compared += 1
    return {
        "v12_c0_minus_v11_c1_max_abs": retrain,
        "control_minus_replication_max_abs": control,
        "replication_queries_compared": compared,
    }


def score(root):
    root = Path(root).resolve()
    config, manifest, _ = cohort(root)
    out = root / "raw" / "eval_v4"
    seal_file = read(out / "prediction_seal.json")
    rows, access = [], []
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        frames = {f["frame_id"]: f for f in record["frames"]}
        for role in ("A", "B"):
            for qid in record["roles"]["primary_query"]:
                entry = seal_file["predictions"][f"{sid}/{role}/{qid}"]
                if sha(Path(entry["path"])) != entry["sha256"]:
                    raise PermissionError("A sealed prediction changed")
                arrays = dict(np.load(entry["path"]))
                access.append(
                    {"scene_id": sid, "frame_id": qid, "purpose": "POST_SEAL_QUERY_DEPTH"}
                )
                gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                valid = np.isfinite(gt) & (gt > 0)
                near = arrays["near"].astype(bool)
                masks = {"ALL": valid, "NEAR": valid & near, "FAR": valid & ~near}
                common = {"scene_id": sid, "role": role, "query_id": qid}
                predictions = {
                    (name, 0): arrays[name]
                    for name in ("REPROJ_NN", "REPROJ_HARMONIC", "REPROJ_TELEA")
                }
                for method in (CONTROL, *PRIMARY_PAIR):
                    for seed in config["seeds"]:
                        carrier = arrays[f"{method}_{seed}"]
                        predictions[method, seed] = carrier
                        predictions[f"HFILL8_{method}", seed] = np.where(
                            near, arrays["REPROJ_HARMONIC"], carrier
                        )
                        predictions[f"FILL8_{method}", seed] = np.where(
                            near, arrays["REPROJ_NN"], carrier
                        )
                for (method, seed), value in predictions.items():
                    row = {**common, "method": method, "seed": seed}
                    for region, mask in masks.items():
                        row[f"absrel_{region}"] = (
                            float(np.mean(np.abs(value[mask] - gt[mask]) / gt[mask]))
                            if mask.any()
                            else None
                        )
                    row["depth_absrel"] = depth_metrics(value, gt)["depth_absrel"]
                    row["far_fraction"] = float(masks["FAR"].sum() / max(valid.sum(), 1))
                    rows.append(row)
    write_json(out / "query_rows.json", rows)
    write_json(out / "GT_access_after_seal.json", access)
    print("EVAL_V4_SCORED", len(rows), flush=True)


def analyze_rows(rows):
    v11, label = host(), inpainting().geometry().label
    methods = sorted({r["method"] for r in rows})

    def values(method, region):
        return v11.scene_values(rows, method, f"absrel_{region}")

    absolute = {m: {region: v11.summary(values(m, region)) for region in REGIONS} for m in methods}
    completion = v11.contrast(values(CONTROL, "FAR"), values("C1", "FAR"))
    harmonic = v11.contrast(values("REPROJ_HARMONIC", "ALL"), values("HFILL8_C1", "ALL"))
    hybrid = v11.contrast(values(f"HFILL8_{CONTROL}", "ALL"), values("HFILL8_C1", "ALL"))
    statuses, checks = {}, {}
    for name, gain in (("WIDTH64_COMPLETION_STATUS", completion), ("HFILL8_WIDTH_STATUS", hybrid)):
        statuses[name], checks[name] = v11.gate(gain) if gain is not None else ("UNDEFINED", None)
    statuses["HARMONIC_HYBRID_LABEL"] = label(harmonic) if harmonic is not None else "UNDEFINED"
    first = statuses["WIDTH64_COMPLETION_STATUS"] == "SUPPORTED"
    second = statuses["HARMONIC_HYBRID_LABEL"] == "ABOVE"
    branch = "A" if first and second else "B" if first or second else "C"
    descriptive = {
        name: v11.contrast(values(a, ra), values(b, rb))
        for name, (a, ra, b, rb) in {
            "V11C1_MINUS_C1_ALL": (CONTROL, "ALL", "C1", "ALL"),
            "V11C1_MINUS_C1_NEAR": (CONTROL, "NEAR", "C1", "NEAR"),
            "C0_MINUS_C1_FAR": ("C0", "FAR", "C1", "FAR"),
            "HARMONIC_MINUS_C1_FAR": ("REPROJ_HARMONIC", "FAR", "C1", "FAR"),
            "HARMONIC_MINUS_HFILL8_V11C1_ALL": (
                "REPROJ_HARMONIC",
                "ALL",
                f"HFILL8_{CONTROL}",
                "ALL",
            ),
            "NN_MINUS_FILL8_C1_ALL": ("REPROJ_NN", "ALL", "FILL8_C1", "ALL"),
        }.items()
    }
    return {
        "absolute_absrel": absolute,
        "WIDTH64_COMPLETION_GAIN": completion,
        "HARMONIC_HYBRID_GAIN": harmonic,
        "HFILL8_WIDTH_GAIN": hybrid,
        "statuses": statuses,
        "gate_checks": checks,
        "descriptive_contrasts": descriptive,
        "INTERPRETATION_BRANCH": branch,
        "far_pixel_fraction": float(
            np.mean([r["far_fraction"] for r in rows if r["method"] == "REPROJ_NN"])
        ),
        "n_scenes": len({r["scene_id"] for r in rows}),
        "definitions": {
            "WIDTH64_COMPLETION_GAIN": "FAR AbsRel(V11 C1 selected) - FAR AbsRel(V12 C1 width 64)",
            "HARMONIC_HYBRID_GAIN": "AbsRel(REPROJ_HARMONIC) - AbsRel(HFILL8 of V12 C1)",
            "HFILL8_WIDTH_GAIN": "AbsRel(HFILL8 of V11 C1) - AbsRel(HFILL8 of V12 C1)",
            "FAR": "valid pixels farther than 8 px from any context-depth reprojection hit",
            "HFILL8": "REPROJ_HARMONIC within 8 px of a hit, the carrier elsewhere",
        },
    }


def analyze(root, output=None):
    root = Path(root).resolve()
    rows_path = root / "raw" / "eval_v4" / "query_rows.json"
    result = analyze_rows(read(rows_path))
    result["consistency"] = read(root / "raw" / "eval_v4" / "prediction_seal.json")["consistency"]
    result["input_sha256"] = {str(rows_path): sha(rows_path)}
    result["source_sha256"] = {
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    target = Path(output) if output else root / "eval_v4_results.json"
    write_json(target, result)
    print(
        "EVAL_V4_BRANCH",
        result["INTERPRETATION_BRANCH"],
        json.dumps(result["statuses"]),
        flush=True,
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
