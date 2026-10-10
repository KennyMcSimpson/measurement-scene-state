"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1: model-free predictions of every episode, from the frozen project functions
only (measurement_planning.geometry: context_points, the 1-pixel z-buffer, the NEAR mask, the harmonic fill).

Per episode and query camera: REPROJ_HARMONIC of the three context frames and its NEAR mask (replayed against the
arms seal of the same cohort: max |difference| <= 1e-9 and identical NEAR, else the stage fails); REPROJ7_HARMONIC
of the three context and four stream frames and its NEAR mask (NEAR7); and the frozen ray cosine of the query
camera (DepthSplat renders planar depth; ray distance = planar / cosine). No query depth is read. Every file is
hashed into geometry_seal.json. Run through cv_host_run.py from the repository root (CUDA hidden):
  python geometry_seal.py ROOT COHORT ARMS_SEAL [--workers 16]
"""

import argparse
import hashlib
import importlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from mcss.measurement_planning import geometry as g
from mcss.measurement_planning.information import TeacherAccess

REPO = Path.cwd()
IMAGE_SIZE = (128, 160)
REPLAY_TOLERANCE = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(job):
    torch.set_num_threads(1)
    record, episode, constant, out = job["record"], job["episode"], job["constant"], Path(job["out"])
    log = []
    teacher = TeacherAccess(record, episode["queries"], IMAGE_SIZE, REPO, log)
    context, stream = episode["context"], episode["stream"]
    points = {**g.frame_points(teacher.frames(context)), **g.frame_points(teacher.frames(stream))}
    results = []
    for qid in episode["queries"]:
        camera = teacher.query_camera(qid)
        cosine = g.ray_cosine(camera)
        maps = {fid: g.reproject(points[fid], camera) for fid in [*context, *stream]}
        base = g.union([maps[f] for f in context])
        base7 = g.union([base, *[maps[f] for f in stream]])
        h3, hit3 = g.harmonic(base, cosine, constant)
        h7, hit7 = g.harmonic(base7, cosine, constant)
        arrays = {"REPROJ_HARMONIC": h3, "near": g.near_mask(hit3), "REPROJ7_HARMONIC": h7,
                  "near7": g.near_mask(hit7), "cosine": np.asarray(cosine, dtype=np.float64)}
        path = out / f"{episode['scene_id']}_{episode['role']}_{qid}.npz"
        np.savez_compressed(path, **arrays)
        results.append({"key": f"{episode['scene_id']}/{episode['role']}/{qid}", "path": str(path), "sha256": sha(path),
                        "hit3": float(hit3.mean()), "hit7": float(hit7.mean())})
    purposes = sorted({e["purpose"] for e in log})
    return results, purposes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("cohort")
    parser.add_argument("arms_seal", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][args.cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    seal = json.loads(args.arms_seal.read_text())
    if seal.get("query_depth_read") is not False:
        raise PermissionError("arms seal does not certify that no query depth was read")
    constant = seal["reprojection_fill_constant_m"]
    out = root / "geometry" / args.cohort
    out.mkdir(parents=True, exist_ok=False)
    jobs = [{"record": records[e["scene_id"]], "episode": e, "constant": constant, "out": str(out)}
            for e in lock["episodes"]]
    # cv_host_run executes this file in a fresh namespace, so workers get the function from the imported module.
    worker = importlib.import_module(Path(__file__).stem)
    with Pool(args.workers) as pool:
        done = pool.map(worker.run, jobs, chunksize=1)
    predictions, purposes, replay = {}, set(), {"max_abs_diff": 0.0, "near_mismatch": [], "missing": []}
    prefix = seal["cohort"]
    for results, used in done:
        purposes |= set(used)
        for item in results:
            predictions[item["key"]] = {k: item[k] for k in ("path", "sha256", "hit3", "hit7")}
            sealed = seal["predictions"].get(f"{prefix}/{item['key']}")
            if sealed is None:
                replay["missing"].append(item["key"])
                continue
            if sha(sealed["path"]) != sealed["sha256"]:
                raise PermissionError(f"sealed arms prediction changed: {item['key']}")
            ours, theirs = np.load(item["path"]), np.load(sealed["path"])
            diff = float(np.max(np.abs(ours["REPROJ_HARMONIC"] - theirs["REPROJ_HARMONIC"])))
            replay["max_abs_diff"] = max(replay["max_abs_diff"], diff)
            if not np.array_equal(ours["near"], theirs["near"].astype(bool)):
                replay["near_mismatch"].append(item["key"])
    replay["pass"] = (replay["max_abs_diff"] <= REPLAY_TOLERANCE and not replay["near_mismatch"]
                      and not replay["missing"])
    if purposes - {"ORACLE_TEACHER_ONLY", "QUERY_CAMERA_TEACHER_ONLY"}:
        raise PermissionError(f"unexpected read purposes {sorted(purposes)}")
    report = {"cohort": args.cohort, "episodes": len(jobs), "queries": len(predictions), "query_depth_read": False,
              "read_purposes": sorted(purposes), "arms_seal": str(args.arms_seal), "arms_seal_sha256": sha(args.arms_seal),
              "reprojection_fill_constant_m": constant, "frozen_sources": g.frozen_hashes(),
              "replay_vs_arms_seal": replay, "predictions": predictions}
    (root / "geometry" / f"geometry_seal_{args.cohort}.json").write_text(json.dumps(report, indent=1))
    print("GEOMETRY_SEALED", args.cohort, len(predictions), "queries; replay", replay["pass"], replay["max_abs_diff"])
    if not replay["pass"]:
        raise SystemExit("REPLAY_FAILED")


if __name__ == "__main__":
    main()
