#!/usr/bin/env python3
"""V11 EVAL-V3 stage: every prediction is sealed to disk before any query depth is read.

1. seal: for each EVAL-V3 scene and role, the context-only RGB-D loader (context frames only)
   gives the context and its CONTEXT_DEPTH bounds; every selected C0/C1 checkpoint builds its
   state and renders every primary query camera; the context-depth reprojection (1-px z-buffer,
   nearest-neighbour fill with the TRAIN constant) and its hit mask come from the same context.
   Every prediction file is hashed. Only query cameras are read.
2. score: query depth is read only after the seal file exists; AbsRel on ALL, NEAR (<= near_px,
   from a reprojection hit) and FAR pixels for the carriers, REPROJ_NN and FILL8 (NEAR from
   REPROJ_NN, FAR from the carrier).
3. analyze: per-scene values (seeds equal, then the mean of role means of queries) and the two
   preregistered primary gates with the frozen V2 gate checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from mcss.dynamic.types import OnlineObservation, hash_scene_state
from mcss.geometry import transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.direct_capacity_contracts import ContextOnlyLoader
from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.geometry_carrier_experiment import read, verify_lock
from mcss.mechanism_pilot.geometry_carrier_statistics import _evidence, surface_status
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_completion_carrier import PRIMARY_PAIR, build_state, load_checkpoint
from mcss.mechanism_pilot.rgbd_depth_bounds import context_depth_bounds
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras

REGIONS = ("ALL", "NEAR", "FAR")
DRAWS, SEED = 10000, 20260928
GEO = Path(__file__).with_name("run_rgbd_geometric_baselines.py")


def geometry():
    spec = importlib.util.spec_from_file_location("v11_geometry", GEO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort(root):
    """The verified EVAL-V3 manifest frozen by prepare (hash-checked)."""
    config = read(root / "training_contract.json")
    entry = config["eval_v3"]
    path = Path(entry["manifest"])
    if sha(path) != entry["manifest_sha256"]:
        raise PermissionError("EVAL-V3 manifest changed after the freeze")
    manifest = read(path)
    if any(r["split"] != "EVAL_V3" for r in manifest["scenes"]):
        raise PermissionError("EVAL-V3 records only")
    return config, manifest, path.parent


def context_of(record, manifest, base, access):
    ids = [r["scene_id"] for r in manifest["scenes"]]
    loader = ContextOnlyLoader(
        record,
        manifest["image_size"],
        base,
        "cpu",
        track="RGBD",
        capacity_scene_ids=ids,
        holdout_scene_ids=manifest.get("protected_scene_ids", []),
        access_log=access,
    )
    contexts = {}
    for role in ("A", "B"):
        batch = loader.context(role)
        observations = [
            OnlineObservation(
                batch.scene_id,
                fid,
                batch.rgb[0, j],
                Cameras(
                    batch.cameras.intrinsics[0, j],
                    batch.cameras.c2w[0, j],
                    batch.cameras.image_size,
                ),
            )
            for j, fid in enumerate(batch.frame_ids)
        ]
        contexts[role] = v5.RGBDContext(observations, batch.depth[0, :, 0])
    return contexts


@torch.no_grad()
def seal(root):
    root = Path(root).resolve()
    verify_lock(root)
    config, manifest, base = cohort(root)
    out = root / "raw" / "eval_v3"
    if (out / "prediction_seal.json").exists():
        raise FileExistsError("EVAL-V3 predictions are sealed once")
    geo = geometry()
    selected = read(root / "selected_checkpoints.json")
    carriers = {}
    for variant in PRIMARY_PAIR:
        for seed in config["seeds"]:
            item = selected[variant][str(seed)]
            if sha(Path(item["path"])) != item["sha256"]:
                raise PermissionError("Selected checkpoint changed")
            model, _ = load_checkpoint(item["path"], "cpu")
            carriers[variant, seed] = model.eval()
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    constant = config["eval_v3"]["reprojection_fill_constant_m"]
    near_px = config["eval_v3"]["near_px"]
    access, entries, states = [], {}, {}
    for record in manifest["scenes"]:
        sid = record["scene_id"]
        contexts = context_of(record, manifest, base, access)
        for role, context in contexts.items():
            anchor = context[0].camera.c2w
            bounds = context_depth_bounds(context, anchor)
            points = geo.context_points(context)
            built = {}
            for (variant, seed), carrier in carriers.items():
                state, state_anchor = build_state(carrier, context, bounds, f"eval_v3:{sid}:{role}")
                built[variant, seed] = (state, state_anchor)
                states[f"{variant}:{seed}:{sid}/{role}"] = hash_scene_state(state)
            for qid in record["roles"]["primary_query"]:
                access.append({"scene_id": sid, "frame_id": qid, "purpose": "QUERY_CAMERA_ONLY"})
                camera = geo.query_camera(record, qid, manifest["image_size"])
                depth, hit = geo.reproject(points, camera)
                reproj = geo.fill_nearest(depth, hit, constant) if hit.any() else None
                if reproj is None:
                    reproj = np.full(camera.image_size, constant, dtype=np.float64)
                near = (
                    ndimage.distance_transform_edt(~hit) <= near_px
                    if hit.any()
                    else np.zeros(camera.image_size, dtype=bool)
                )
                arrays = {"REPROJ_NN": reproj, "hit": hit, "near": near}
                batched = Cameras(
                    camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
                )
                for (variant, seed), (state, state_anchor) in built.items():
                    local = transform_cameras(batched, torch.linalg.inv(state_anchor))
                    pred = renderer(state, local, ("depth",))["depth"][0, 0, 0].numpy()
                    arrays[f"{variant}_{seed}"] = pred.astype(np.float64)
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
            "near_px": near_px,
            "reprojection_fill_constant_m": constant,
        },
    )
    print("EVAL_V3_PREDICTIONS_SEALED", len(entries), flush=True)


def score(root):
    root = Path(root).resolve()
    config, manifest, _ = cohort(root)
    out = root / "raw" / "eval_v3"
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
                masks = {
                    "ALL": valid,
                    "NEAR": valid & arrays["near"],
                    "FAR": valid & ~arrays["near"],
                }
                common = {"scene_id": sid, "role": role, "query_id": qid}
                predictions = {("REPROJ_NN", 0): arrays["REPROJ_NN"]}
                for variant in PRIMARY_PAIR:
                    for seed in config["seeds"]:
                        carrier = arrays[f"{variant}_{seed}"]
                        predictions[variant, seed] = carrier
                        predictions[f"FILL8_{variant}", seed] = np.where(
                            arrays["near"], arrays["REPROJ_NN"], carrier
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
    print("EVAL_V3_SCORED", len(rows), flush=True)


def scene_values(rows, method, field):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] == method and r[field] is not None:
            grouped[r["seed"], r["scene_id"]][r["role"]].append(r[field])
    per_scene = defaultdict(list)
    for (_, sid), roles in grouped.items():
        per_scene[sid].append(float(np.mean([np.mean(v) for v in roles.values()])))
    return {sid: float(np.mean(v)) for sid, v in sorted(per_scene.items())}


def gate(gain):
    checks = {
        "mean_ci_and_scene_consistency": _evidence(gain),
        "every_leave_one_scene_out_gain_positive": bool(gain["loso"])
        and min(gain["loso"].values()) > 0,
        "positive_top1_share_at_most_0_5": gain["positive_top1_contribution"] is not None
        and gain["positive_top1_contribution"] <= 0.5,
    }
    return surface_status(gain, checks), checks


def summary(values):
    """Scene bootstrap description; None when the region holds no pixel in any scene."""
    return describe(values, draws=DRAWS, seed=SEED) if values else None


def contrast(a, b):
    """Paired scene bootstrap of a - b over their common scenes; None when there is none."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    return paired({s: a[s] for s in common}, {s: b[s] for s in common}, draws=DRAWS, seed=SEED)


def analyze_rows(rows):
    methods = sorted({r["method"] for r in rows})
    fields = {region: f"absrel_{region}" for region in REGIONS}
    absolute = {
        m: {region: summary(scene_values(rows, m, f)) for region, f in fields.items()}
        for m in methods
    }
    completion = contrast(
        scene_values(rows, "C0", "absrel_FAR"), scene_values(rows, "C1", "absrel_FAR")
    )
    hybrid = contrast(
        scene_values(rows, "REPROJ_NN", "absrel_ALL"), scene_values(rows, "FILL8_C1", "absrel_ALL")
    )
    statuses, checks = {}, {}
    for name, gain in (("COMPLETION_STATUS", completion), ("HYBRID_STATUS", hybrid)):
        statuses[name], checks[name] = gate(gain) if gain is not None else ("UNDEFINED", None)
    supported = [statuses[k] == "SUPPORTED" for k in ("COMPLETION_STATUS", "HYBRID_STATUS")]
    branch = "A" if all(supported) else "B" if any(supported) else "C"
    descriptive = {
        name: contrast(scene_values(rows, a, fa), scene_values(rows, b, fb))
        for name, (a, fa, b, fb) in {
            "C0_MINUS_C1_ALL": ("C0", "absrel_ALL", "C1", "absrel_ALL"),
            "C0_MINUS_C1_NEAR": ("C0", "absrel_NEAR", "C1", "absrel_NEAR"),
            "REPROJ_MINUS_C1_FAR": ("REPROJ_NN", "absrel_FAR", "C1", "absrel_FAR"),
            "REPROJ_MINUS_FILL8_C0_ALL": ("REPROJ_NN", "absrel_ALL", "FILL8_C0", "absrel_ALL"),
            "REPROJ_MINUS_C1_ALL": ("REPROJ_NN", "absrel_ALL", "C1", "absrel_ALL"),
        }.items()
    }
    return {
        "absolute_absrel": absolute,
        "COMPLETION_GAIN": completion,
        "HYBRID_GAIN": hybrid,
        "statuses": statuses,
        "gate_checks": checks,
        "descriptive_contrasts": descriptive,
        "INTERPRETATION_BRANCH": branch,
        "far_pixel_fraction": float(
            np.mean([r["far_fraction"] for r in rows if r["method"] == "REPROJ_NN"])
        ),
        "n_scenes": len({r["scene_id"] for r in rows}),
        "definitions": {
            "COMPLETION_GAIN": "FAR AbsRel(C0) - FAR AbsRel(C1)",
            "HYBRID_GAIN": "AbsRel(REPROJ_NN) - AbsRel(FILL8 of C1), all valid pixels",
            "FAR": "valid pixels farther than near_px (training contract) from any context-depth "
            "reprojection hit",
            "FILL8": "REPROJ_NN within near_px of a hit, the carrier elsewhere",
        },
    }


def analyze(root, output=None):
    root = Path(root).resolve()
    rows_path = root / "raw" / "eval_v3" / "query_rows.json"
    result = analyze_rows(read(rows_path))
    result["input_sha256"] = {str(rows_path): sha(rows_path)}
    result["source_sha256"] = {
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    target = Path(output) if output else root / "eval_v3_results.json"
    write_json(target, result)
    print(
        "EVAL_V3_BRANCH",
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
    {"seal": seal, "score": score}.get(args.stage, lambda root: analyze(root, args.output))(
        args.root
    )
