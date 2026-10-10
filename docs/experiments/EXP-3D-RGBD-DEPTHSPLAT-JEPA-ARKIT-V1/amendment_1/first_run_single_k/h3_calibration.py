"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1, amendment 1 (written 2026-10-09, before this experiment reads any ARKIT160
query depth): a calibrated harmonic baseline, fitted on the ARKit TRAINING records only.

Why: on ARKit the harmonic fill is biased far in FAR (ENSEMBLES-ARKIT bias check: median log(H3 / gt) +0.10 with a
heavy over-estimation tail; FAR AbsRel 0.75-0.84). A model trained on ARKit can learn a global shrink of unseen
depth. To tell such a shrink from completion information, H3CAL multiplies H3 by one constant k outside NEAR; k
minimises the scene-mean FAR AbsRel of k x H3 over the TRAIN records of the training data (episodes = record x role
A/B, context = the role frames, queries = primary_query: the ARKIT160 rule) on the grid 0.40, 0.45, ..., 1.20.
Only training data are read (TRAIN query depth). Also writes descriptive diagnostics of the FAR bias on TRAIN: the
depth of FAR pixels relative to the nearest hit, by distance to the nearest hit and by image row.
Run through cv_host_run.py from the repository root (CUDA hidden):
  python h3_calibration.py ROOT [--workers 8]
"""
import argparse
import hashlib
import importlib
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from mcss.measurement_planning import geometry as g
from mcss.measurement_planning.information import TeacherAccess

REPO = Path.cwd()
TRAINING = REPO / "outputs/EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json"
IMAGE_SIZE = (128, 160)
GRID = [round(0.40 + 0.05 * i, 2) for i in range(17)]
DIST_BINS = ((8, 16), (16, 32), (32, 64), (64, 100000))
ROWS = ("top", "middle", "bottom")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def median_log(x):
    return float(np.median(np.log(x))) if x.size else None


def run(job):
    torch.set_num_threads(1)
    record, role, context, constant = job["record"], job["role"], job["context"], job["constant"]
    queries = list(record["roles"]["primary_query"])
    teacher = TeacherAccess(record, queries, IMAGE_SIZE, REPO, [])
    points = g.frame_points(teacher.frames(context))
    frames = {f["frame_id"]: f for f in record["frames"]}
    out = []
    for qid in queries:
        camera = teacher.query_camera(qid)
        cosine = g.ray_cosine(camera)
        h3, hit = g.harmonic(g.union([g.reproject(points[f], camera) for f in context]), cosine, constant)
        near = np.asarray(g.near_mask(hit), dtype=bool)
        gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)  # TRAINING data only
        valid = np.isfinite(gt) & (gt > 0)
        far = valid & ~near
        row = {"scene_id": record["scene_id"], "role": role, "query": qid, "far_px": int(far.sum()),
               "valid_px": int(valid.sum()), "hit_share": float(hit.mean())}
        if far.any():
            ratio = h3[far] / gt[far]
            row["absrel_k"] = [float(np.mean(np.abs(k * ratio - 1.0))) for k in GRID]
            row["log_h3_gt"] = median_log(ratio)
            # the nearest hit (its ray depth, kept unchanged by the harmonic fill) and the distance to it
            dist, (ii, jj) = ndimage.distance_transform_edt(~hit, return_indices=True)
            nearest = h3[ii, jj][far]
            row["log_gt_nearest_hit"] = median_log(gt[far] / nearest)
            row["share_gt_below_0.8_nearest_hit"] = float(np.mean(gt[far] < 0.8 * nearest))
            row["share_gt_above_1.25_nearest_hit"] = float(np.mean(gt[far] > 1.25 * nearest))
            row["share_h3_over_2gt"] = float(np.mean(ratio > 2.0))
            row["absrel_part_h3_over_2gt"] = float(np.sum(np.abs(ratio - 1.0) * (ratio > 2.0)) / far.sum())
            row["median_gt_far"] = float(np.median(gt[far]))
            row["median_gt_valid"] = float(np.median(gt[valid]))
            d = dist[far]
            row["by_distance"] = {f"{lo}-{hi}": {"share": float(np.mean((d >= lo) & (d < hi))),
                                               "log_h3_gt": median_log(ratio[(d >= lo) & (d < hi)])}
                                  for lo, hi in DIST_BINS}
            third = np.minimum(np.nonzero(far)[0] * 3 // IMAGE_SIZE[0], 2)
            row["by_row"] = {name: {"share": float(np.mean(third == t)), "log_h3_gt": median_log(ratio[third == t])}
                             for t, name in enumerate(ROWS)}
        out.append(row)
    return out


def scene_mean(rows, value):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        v = value(row)
        if v is not None:
            grouped[row["scene_id"]][row["role"]].append(v)
    scenes = {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in grouped.items()}
    return float(np.mean(list(scenes.values()))), len(scenes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / "evaluation_contract.json").read_text())
    prereg = json.loads((root / "preregistration.json").read_text())
    if sha(TRAINING) != prereg["training_data"]["manifest_arkit_trainval.json"]:
        raise PermissionError("training manifest changed since the lock")
    records = [r for r in json.loads(TRAINING.read_text())["scenes"] if r["split"] == "TRAIN"]
    evaluation = json.loads(Path(contract["cohorts"]["ARKIT160"]["manifest"]).read_text())["scenes"]
    if {r["physical_scene_id"] for r in records} & {r["physical_scene_id"] for r in evaluation}:
        raise ValueError("a TRAIN venue is in ARKIT160")
    jobs = [{"record": r, "role": role, "context": list(r["roles"][key]),
             "constant": contract["reprojection_fill_constant_m"]}
            for r in records for role, key in (("A", "context_a"), ("B", "context_b"))]
    # cv_host_run executes this file in a fresh namespace, so workers get the function from the imported module.
    worker = importlib.import_module(Path(__file__).stem)
    with Pool(args.workers) as pool:
        rows = [row for part in pool.map(worker.run, jobs, chunksize=1) for row in part]
    used = [r for r in rows if "absrel_k" in r]
    curve = {f"{k:.2f}": scene_mean(used, lambda r, i=i: r["absrel_k"][i])[0] for i, k in enumerate(GRID)}
    k_star = float(min(GRID, key=lambda k: (round(curve[f"{k:.2f}"], 12), abs(k - 1.0))))
    scalar = ("log_h3_gt", "log_gt_nearest_hit", "share_gt_below_0.8_nearest_hit", "share_gt_above_1.25_nearest_hit",
              "share_h3_over_2gt", "absrel_part_h3_over_2gt", "median_gt_far", "median_gt_valid", "hit_share")
    diagnostics = {name: scene_mean(used, lambda r, n=name: r[n])[0] for name in scalar}
    for group, keys in (("by_distance", [f"{lo}-{hi}" for lo, hi in DIST_BINS]), ("by_row", ROWS)):
        diagnostics[group] = {key: {field: scene_mean(used, lambda r, g_=group, k_=key, f_=field: r[g_][k_][f_])[0]
                                    for field in ("share", "log_h3_gt")} for key in keys}
    far_absrel = np.array([r["absrel_k"][GRID.index(1.0)] for r in used])
    gt_med = np.array([r["median_gt_far"] for r in used])
    diagnostics["spearman_query_far_absrel_vs_median_gt_far"] = float(
        np.corrcoef(np.argsort(np.argsort(far_absrel)), np.argsort(np.argsort(gt_med)))[0, 1])
    report = {"experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1", "amendment": 1, "data": "TRAIN (training data only)",
              "training_manifest_sha256": sha(TRAINING), "records": len(records), "queries": len(rows),
              "queries_with_far": len(used), "scenes": scene_mean(used, lambda r: r["absrel_k"][0])[1],
              "grid": GRID, "scene_mean_far_absrel_by_k": curve, "k_star": k_star,
              "fill_constant_m": contract["reprojection_fill_constant_m"],
              "frozen_sources": g.frozen_hashes(), "diagnostics": diagnostics, "rows": rows}
    out = root / "amendment_1"
    out.mkdir(exist_ok=True)
    (out / "h3_calibration_TRAIN.json").write_text(json.dumps(report, indent=1))
    print("H3_CALIBRATED k_star", k_star, "scenes", report["scenes"], "queries", len(used))
    print("  FAR AbsRel by k:", {k: round(v, 4) for k, v in curve.items()})
    print("  diagnostics:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in diagnostics.items()}, default=str))


if __name__ == "__main__":
    main()
