#!/usr/bin/env python3
"""Strong non-learned geometric baselines for the qualified RGB-D carrier (no training).

Every baseline sees exactly the carrier's test-time input for one episode: the role's 3
context frames (RGB, measured ray distance, camera) and the query camera. REPROJ_NN (primary)
reprojects every valid context depth sample to the query camera with a 1-pixel z-buffer and
fills empty pixels from the nearest filled pixel; REPROJ_CONST fills them with the TRAIN
AbsRel-optimal constant; NLF16_DEPTH is the non-learned voxel fusion in the same 16^3 grid,
bounds rule and renderer as the carrier (four scalars fit on TRAIN24, exploratory diagnostic).
HYBRID (descriptive) takes REPROJ where a context sample landed and the carrier elsewhere.

`--stage lock` freezes inputs before any baseline is computed; `--stage run` computes and seals
every baseline prediction for DEV and FRESH-V2 before any query GT is read, then scores the
baselines and the sealed V7 C1 predictions with one metric function.
"""
# ruff: noqa: E501 -- Chinese scientific report prose

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from mcss.geometry import generate_rays, project_world, transform_cameras
from mcss.measurements import FixedMeasurementRenderer
from mcss.mechanism_pilot.direct_capacity_statistics import describe, paired
from mcss.mechanism_pilot.geometry_carrier_statistics import _evidence
from mcss.mechanism_pilot.readout_reanalysis import depth_metrics
from mcss.mechanism_pilot.rgbd_depth_bounds import DepthBoundsSceneData
from mcss.mechanism_pilot.rgbd_fresh_v2_evaluation import FreshDepthBoundsSceneData
from mcss.mechanism_pilot.rgbd_nonlearned_fusion import fused_state
from mcss.types import Cameras

EXPERIMENT = "EXP-3D-RGBD-GEOMETRIC-BASELINES-V1"
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
QUALIFICATION = Path("outputs/EXP-3D-RGBD-FRESH-QUALIFICATION-V2")
NLF_FIT = Path(
    "outputs/EXP-3D-RGBD-RESOLUTION-V8/audit/exploratory_nlf_resolution_train_only/"
    "claude_nlf_resolution_CONTEXT_DEPTH_16.json"
)
BASELINES = ("REPROJ_NN", "REPROJ_CONST", "NLF16_DEPTH")
PRIMARY = "REPROJ_NN"
CARRIER = "C1"
DRAWS, SEED = 10000, 20260928


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def cohorts(v7=V7, qualification=QUALIFICATION):
    return {
        "DEV": {
            "manifest": v7 / "scene_split.json",
            "split": "DEV",
            "root": v7,
            "carrier": v7 / "raw/selected_dev",
            "observability": v7 / "raw/observability",
            "loader": DepthBoundsSceneData,
        },
        "FRESH_V2": {
            "manifest": qualification / "scene_split.json",
            "split": "FRESH_QUALIFICATION",
            "root": qualification,
            "carrier": qualification / "raw/selected_fresh",
            "observability": qualification / "raw/observability",
            "loader": FreshDepthBoundsSceneData,
        },
    }


# ---------------------------------------------------------------- geometry (no learned part)
def context_points(context):
    """World points of every valid measured context depth sample [N, 3] (float64)."""
    points = []
    for observation, depth in zip(context, context.depths, strict=True):
        camera = observation.camera.to(dtype=torch.float64)
        origins, rays = generate_rays(camera)
        distance = torch.as_tensor(depth, dtype=torch.float64)
        valid = torch.isfinite(distance) & (distance > 0)
        points.append(origins[valid] + rays[valid] * distance[valid, None])
    return torch.cat(points)


def reproject(points, camera):
    """1-pixel z-buffer of ray distance at the query camera: (distance [H, W], hit [H, W])."""
    camera = camera.to(dtype=torch.float64)
    height, width = camera.image_size
    pixels, _, inside = project_world(points, camera)
    u = pixels[:, 0].round().long().clamp(0, width - 1)
    v = pixels[:, 1].round().long().clamp(0, height - 1)
    distance = (points - camera.c2w[:3, 3]).norm(dim=-1)
    flat = torch.full((height * width,), float("inf"), dtype=torch.float64)
    flat.scatter_reduce_(0, (v * width + u)[inside], distance[inside], reduce="amin")
    depth = flat.reshape(height, width).numpy()
    return depth, np.isfinite(depth)


def fill_nearest(depth, hit, constant):
    if not hit.any():
        return np.full(depth.shape, constant, dtype=np.float64)
    _, (iy, ix) = ndimage.distance_transform_edt(~hit, return_indices=True)
    return depth[iy, ix]


def fill_constant(depth, hit, constant):
    return np.where(hit, depth, constant)


def query_camera(record, fid, image_size):
    frame = next(f for f in record["frames"] if f["frame_id"] == fid)
    return Cameras(
        torch.tensor(frame["intrinsics"], dtype=torch.float32),
        torch.tensor(frame["c2w"], dtype=torch.float32),
        tuple(image_size),
    )


def label(stats):
    """ABOVE: the carrier is better with frozen V2 evidence; BELOW: the baseline is."""
    if _evidence(stats):
        return "ABOVE"
    lo, hi = stats["ci95"]
    mirror = {
        **stats,
        "mean": -stats["mean"],
        "ci95": [-hi, -lo],
        "positive": stats["negative"],
        "negative": stats["positive"],
    }
    return "BELOW" if _evidence(mirror) else "NOT_DISTINGUISHABLE"


# ---------------------------------------------------------------- stages
def lock(root, v7=V7, qualification=QUALIFICATION, nlf_fit=NLF_FIT):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact experiment directory required")
    if (root / "lock.json").exists():
        raise FileExistsError("Never re-lock")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Protocol text required before locking")
    if read(v7 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V7 must be finalized")
    fresh = read(qualification / "fresh_qualification_results.json")
    if fresh["FRESH_QUALIFICATION_STATUS"] != "QUALIFIED":
        raise PermissionError("The carrier under test must be the qualified V7 C1")
    fit = read(nlf_fit)
    if fit["rule"] != "CONTEXT_DEPTH" or fit["grid"] != 16:
        raise PermissionError("NLF16 parameters must come from the TRAIN-only depth-bounds fit")
    constant = read(v7 / "decision_rules.json")["GEOMETRY_FREE_REFERENCE"]["values"][
        "REF_TRAIN_ABSREL_OPTIMAL"
    ]
    seeds = read(v7 / "training_contract.json")["seeds"]
    carrier_files = {}
    for cohort in cohorts(v7, qualification).values():
        for seed in seeds:
            directory = cohort["carrier"] / f"{CARRIER}_{seed}"
            for row in read(directory / "query_results.json"):
                if row["method"] == "direct":
                    path = directory / row["prediction_path"]
                    if sha(path) != row["prediction_file_sha256"]:
                        raise PermissionError(f"Sealed carrier prediction changed: {path}")
                    carrier_files[str(path.resolve())] = row["prediction_file_sha256"]
    inputs = [
        root / "PROTOCOL.md",
        v7 / "scene_split.json",
        v7 / "integrity.json",
        v7 / "decision_rules.json",
        qualification / "scene_split.json",
        qualification / "fresh_qualification_results.json",
        nlf_fit,
        Path(__file__).resolve(),
    ]
    write(
        root / "lock.json",
        {
            "status": "FROZEN_BEFORE_BASELINE_COMPUTATION",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "carrier": f"V7 {CARRIER} sealed per-seed selections (qualified Core A)",
            "seeds": seeds,
            "baselines": list(BASELINES),
            "primary": f"{PRIMARY} vs V7 {CARRIER} on DEV",
            "constant_m": constant,
            "nlf16_parameters": fit["best_params"],
            "input_sha256": {str(Path(p).resolve()): sha(p) for p in inputs},
            "carrier_prediction_sha256": carrier_files,
            "baseline_computed_before_lock": False,
            "query_gt_read_before_lock": False,
            "final_holdout_opened": False,
        },
    )
    print("LOCKED", len(carrier_files), "sealed carrier predictions", flush=True)


def verify(root):
    lock_ = read(root / "lock.json")
    for path, digest in {**lock_["input_sha256"], **lock_["carrier_prediction_sha256"]}.items():
        if sha(path) != digest:
            raise PermissionError(f"Locked file changed: {path}")
    return lock_


def compute_baselines(root, lock_, cohort_map, access):
    """Every baseline prediction of every episode, sealed to disk before any query GT."""
    renderer = FixedMeasurementRenderer(n_samples=64, ray_chunk_size=2048)
    sealed = {}
    for name, cohort in cohort_map.items():
        manifest = read(cohort["manifest"])
        records = [r for r in manifest["scenes"] if r["split"] == cohort["split"]]
        for record in records:
            data = cohort["loader"](record, manifest, cohort["root"], "cpu", access)
            for role in ("A", "B"):
                context = data.context(role)
                points = context_points(context)
                with torch.no_grad():
                    state, anchor = fused_state(
                        context, data.bounds[role], lock_["nlf16_parameters"]
                    )
                for qid in record["roles"]["primary_query"]:
                    camera = query_camera(record, qid, manifest["image_size"])
                    access.append(
                        {
                            "scene_id": record["scene_id"],
                            "frame_id": qid,
                            "purpose": "QUERY_CAMERA_ONLY",
                        }
                    )
                    depth, hit = reproject(points, camera)
                    batched = Cameras(
                        camera.intrinsics[None, None], camera.c2w[None, None], camera.image_size
                    )
                    local = transform_cameras(batched, torch.linalg.inv(anchor))
                    with torch.no_grad():
                        nlf = renderer(state, local, ("depth",))["depth"][0, 0, 0].numpy()
                    predictions = {
                        "REPROJ_NN": fill_nearest(depth, hit, lock_["constant_m"]),
                        "REPROJ_CONST": fill_constant(depth, hit, lock_["constant_m"]),
                        "NLF16_DEPTH": nlf.astype(np.float64),
                    }
                    key = f"{name}/{record['scene_id']}/{role}/{qid}"
                    path = (
                        root / "raw" / "baselines" / name / f"{record['scene_id']}_{role}_{qid}.npz"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path, hit=hit, reproj=np.where(hit, depth, 0.0), **predictions
                    )
                    sealed[key] = {
                        "path": str(path),
                        "sha256": sha(path),
                        "hit_fraction": float(hit.mean()),
                    }
    return sealed


def carrier_predictions(cohort, seed):
    directory = cohort["carrier"] / f"{CARRIER}_{seed}"
    rows = [r for r in read(directory / "query_results.json") if r["method"] == "direct"]
    return {
        (r["scene_id"], r["role"], r["query_id"]): (directory / r["prediction_path"], r)
        for r in rows
    }


def aggregate(rows, method, cohort, metric="depth_absrel"):
    """Scene value: seeds equally weighted, then the mean of role means of query values."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] == method and r["cohort"] == cohort:
            grouped[r["seed"], r["scene_id"]][r["role"]].append(r[metric])
    per_scene = defaultdict(list)
    for (_, sid), roles in grouped.items():
        per_scene[sid].append(float(np.mean([np.mean(v) for v in roles.values()])))
    return {sid: float(np.mean(v)) for sid, v in sorted(per_scene.items())}


def region_rows(prediction, gt, obs_class):
    valid = np.isfinite(gt) & (gt > 0)
    out = {}
    for index, region in enumerate(("OBS0", "OBS1", "OBS2PLUS")):
        mask = valid & (obs_class == index)
        out[region] = (
            float(np.mean(np.abs(prediction[mask] - gt[mask]) / gt[mask])) if mask.any() else None
        )
    return out


def run(root, cohort_map=None):
    root = Path(root).resolve()
    lock_ = verify(root)
    cohort_map = cohort_map or cohorts()
    if (root / "raw" / "baselines").exists():
        raise FileExistsError("Baselines already computed; the run happens once")
    access = []
    sealed = compute_baselines(root, lock_, cohort_map, access)
    write(root / "raw" / "baseline_seal.json", sealed)
    access.append({"event": "ALL_BASELINE_PREDICTIONS_SEALED", "count": len(sealed)})
    rows, worst = [], 0.0
    for name, cohort in cohort_map.items():
        manifest = read(cohort["manifest"])
        records = {r["scene_id"]: r for r in manifest["scenes"] if r["split"] == cohort["split"]}
        carriers = {seed: carrier_predictions(cohort, seed) for seed in lock_["seeds"]}
        for sid, record in records.items():
            frames = {f["frame_id"]: f for f in record["frames"]}
            for role in ("A", "B"):
                for qid in record["roles"]["primary_query"]:
                    entry = sealed[f"{name}/{sid}/{role}/{qid}"]
                    if sha(entry["path"]) != entry["sha256"]:
                        raise AssertionError("Sealed baseline changed")
                    baseline = dict(np.load(entry["path"]))
                    access.append(
                        {"scene_id": sid, "frame_id": qid, "purpose": "POST_SEAL_SCORING"}
                    )
                    gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                    obs = np.load(cohort["observability"] / f"{sid}_{role}_{qid}.npz")["obs_class"]
                    common = {"cohort": name, "scene_id": sid, "role": role, "query_id": qid}
                    for method in BASELINES:
                        rows.append(
                            {
                                **common,
                                "seed": 0,
                                "method": method,
                                "hit_fraction": entry["hit_fraction"],
                                **depth_metrics(baseline[method], gt),
                                "regions": region_rows(baseline[method], gt, obs),
                            }
                        )
                    for seed in lock_["seeds"]:
                        path, sealed_row = carriers[seed][sid, role, qid]
                        carrier = np.load(path)["depth"].astype(np.float64)
                        metrics = depth_metrics(carrier, gt)
                        worst = max(
                            worst, abs(metrics["depth_absrel"] - sealed_row["depth_absrel"])
                        )
                        hybrid = np.where(baseline["hit"], baseline["reproj"], carrier)
                        for method, prediction, values in (
                            (f"V7_{CARRIER}", carrier, metrics),
                            ("HYBRID", hybrid, depth_metrics(hybrid, gt)),
                        ):
                            rows.append(
                                {
                                    **common,
                                    "seed": seed,
                                    "method": method,
                                    **values,
                                    "regions": region_rows(prediction, gt, obs),
                                }
                            )
    if worst > 1e-6:
        raise AssertionError(f"Carrier metrics do not reproduce the sealed values: {worst}")
    write(root / "raw" / "scored_rows.json", rows)
    write(root / "audit" / "GT_access.json", access)
    result = analyze(rows, cohort_map)
    result["integrity"] = {
        "max_carrier_absrel_deviation": worst,
        "baseline_predictions_sealed_before_gt": True,
        "sealed_baselines": len(sealed),
        "scored_rows": len(rows),
    }
    write(root / "baseline_results.json", result)
    report(root, result, lock_)
    return result


def analyze(rows, cohort_map):
    carrier = f"V7_{CARRIER}"
    methods = (*BASELINES, "HYBRID", carrier)
    result = {"cohorts": {}}
    for name in cohort_map:
        values = {m: aggregate(rows, m, name) for m in methods}
        comparisons = {}
        for method in (*BASELINES, "HYBRID"):
            stats = paired(values[method], values[carrier], draws=DRAWS, seed=SEED)
            comparisons[method] = {"carrier_minus_baseline_gain": stats, "label": label(stats)}
        regions = {}
        for method in methods:
            per_region = {}
            for region in ("OBS0", "OBS1", "OBS2PLUS"):
                grouped = defaultdict(list)
                for r in rows:
                    if (
                        r["method"] == method
                        and r["cohort"] == name
                        and r["regions"][region] is not None
                    ):
                        grouped[r["scene_id"]].append(r["regions"][region])
                per_region[region] = (
                    float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
                )
            regions[method] = per_region
        hits = [r["hit_fraction"] for r in rows if r["cohort"] == name and r["method"] == PRIMARY]
        result["cohorts"][name] = {
            "absolute_depth_absrel": {
                m: describe(values[m], draws=DRAWS, seed=SEED) for m in methods
            },
            "absolute_depth_delta1": {
                m: float(np.mean(list(aggregate(rows, m, name, "depth_delta1").values())))
                for m in methods
            },
            "comparisons": comparisons,
            "region_depth_absrel": regions,
            "mean_query_hit_fraction": float(np.mean(hits)),
            "n_scenes": len(values[carrier]),
        }
    primary = result["cohorts"]["DEV"]["comparisons"][PRIMARY]["label"]
    result["PRIMARY_LABEL"] = primary
    result["INTERPRETATION_BRANCH"] = {"ABOVE": "A", "NOT_DISTINGUISHABLE": "B", "BELOW": "C"}[
        primary
    ]
    return result


def report(root, result, lock_):
    names = {
        "REPROJ_NN": "上下文深度重投影 + 最近邻填洞（主要基线）",
        "REPROJ_CONST": "上下文深度重投影 + 常数填洞",
        "NLF16_DEPTH": "同网格同 bounds 的非学习体素融合",
        "HYBRID": "重投影命中处用重投影，其余用 carrier（描述性）",
        "V7_C1": "学习式 carrier（已合格的 Core A）",
    }
    lines = ["# EXP-3D-RGBD-GEOMETRIC-BASELINES-V1", ""]
    branch = result["INTERPRETATION_BRANCH"]
    lines.append(
        f"主问题：在同样的测试时输入下，学到的场景状态（V7 C1）是否优于直接的几何重投影（REPROJ_NN）？DEV 标签：{result['PRIMARY_LABEL']}，分支 {branch}。"
    )
    for name, block in result["cohorts"].items():
        lines += [
            "",
            f"## {name}（{block['n_scenes']} 个场景；query 像素平均有 {block['mean_query_hit_fraction']:.3f} 被重投影覆盖）",
            "",
            "| 方法 | 说明 | AbsRel (95% CI) | delta1 | OBS0 / OBS1 / OBS2PLUS AbsRel |",
            "|---|---|---|---|---|",
        ]
        for method, label_text in names.items():
            v = block["absolute_depth_absrel"][method]
            reg = block["region_depth_absrel"][method]
            regions = " / ".join(
                "—" if reg[k] is None else f"{reg[k]:.3f}" for k in ("OBS0", "OBS1", "OBS2PLUS")
            )
            lines.append(
                f"| {method} | {label_text} | {v['mean']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {block['absolute_depth_delta1'][method]:.3f} | {regions} |"
            )
        lines += [
            "",
            "| 对比 | AbsRel(基线) − AbsRel(carrier) | 场景 improved/tied/worse | 标签 |",
            "|---|---|---|---|",
        ]
        for method, comparison in block["comparisons"].items():
            s = comparison["carrier_minus_baseline_gain"]
            lines.append(
                f"| {method} | {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | {s['positive']}/{s['tied']}/{s['negative']} | {comparison['label']} |"
            )
    lines += [
        "",
        {
            "A": "分支 A：学到的状态显著优于直接几何重投影，逆向 JEPA 在 RGB-D 设定下带来超出简单几何的价值。",
            "B": "分支 B：学到的状态与直接几何重投影无法区分；逆向 JEPA 目前只做到与简单几何相当。",
            "C": "分支 C：直接几何重投影显著优于学到的状态；逆向 JEPA 在 RGB-D 设定下尚未带来超出简单几何的价值。",
        }[branch],
        "",
        "FRESH-V2 已在资格验证中用过一次；这里只对不需训练的基线与已封存的 carrier 预测做描述性比较，不改变任何模型或规则。受保护的 final holdout 未打开。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    summary = [
        "RGBD GEOMETRIC BASELINES V1",
        f"PRIMARY_LABEL={result['PRIMARY_LABEL']}",
        f"INTERPRETATION_BRANCH={branch}",
    ]
    for name, block in result["cohorts"].items():
        for method, v in block["absolute_depth_absrel"].items():
            summary.append(f"{name}_{method}_ABSREL={v['mean']:.6f}")
        for method, comparison in block["comparisons"].items():
            s = comparison["carrier_minus_baseline_gain"]
            summary.append(
                f"{name}_C1_VS_{method}={s['mean']:.6f} CI={s['ci95']} LABEL={comparison['label']}"
            )
    summary.append("FINAL_HOLDOUT_TOUCHED=false")
    (root / "terminal_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "lock":
        lock(args.root)
    else:
        run(args.root)


if __name__ == "__main__":
    main()
