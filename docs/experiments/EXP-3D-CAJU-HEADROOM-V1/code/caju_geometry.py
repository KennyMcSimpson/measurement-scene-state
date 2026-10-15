"""EXP-3D-CAJU-HEADROOM-V1: R7 at every query and history camera, from the frozen project functions only (as V1's
geometry_seal.py: measurement_planning.geometry frame_points, the 1-pixel z-buffer, the NEAR mask, the harmonic fill).

Per episode and target camera (queries + history views): REPROJ7_HARMONIC of the 3 context + 4 stream frames, its
NEAR7 mask, the hit share and the frozen ray cosine. The fill constant is V3's (read from V3's geometry seal, whose
hash is recorded). No query or history depth is read. For CAJU_COMPAT the arrays are replayed against V3's sealed
geometry (max |difference| <= 1e-9, identical NEAR7), else the stage fails.
Run through cv_host_run.py from the repository root (CUDA hidden):  python caju_geometry.py ROOT COHORT [--workers 16]
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
V3_SEAL = Path("outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/geometry/geometry_seal_ARKIT_SYSTEM3.json").resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(job):
    torch.set_num_threads(1)
    record, episode, constant, out = job["record"], job["episode"], job["constant"], Path(job["out"])
    log = []
    targets = [*episode["queries"], *episode["history"]]
    teacher = TeacherAccess(record, targets, IMAGE_SIZE, REPO, log)
    inputs = [*episode["context"], *episode["stream"]]
    points = {**g.frame_points(teacher.frames(episode["context"])), **g.frame_points(teacher.frames(episode["stream"]))}
    results = []
    for fid in targets:
        camera = teacher.query_camera(fid)
        cosine = g.ray_cosine(camera)
        base7 = g.union([g.reproject(points[f], camera) for f in inputs])
        r7, hit7 = g.harmonic(base7, cosine, constant)
        arrays = {"REPROJ7_HARMONIC": r7, "near7": g.near_mask(hit7), "cosine": np.asarray(cosine, dtype=np.float64)}
        path = out / f"{episode['scene_id']}_{episode['role']}_{fid}.npz"
        np.savez_compressed(path, **arrays)
        results.append({"key": f"{episode['scene_id']}/{episode['role']}/{fid}", "path": str(path),
                        "sha256": sha(path), "kind": "query" if fid in episode["queries"] else "history",
                        "hit7": float(hit7.mean())})
    return results, sorted({e["purpose"] for e in log})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("cohort")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][args.cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    v3 = json.loads(V3_SEAL.read_text())
    constant = v3["reprojection_fill_constant_m"]
    out = root / "geometry" / args.cohort
    out.mkdir(parents=True, exist_ok=False)
    jobs = [{"record": records[e["scene_id"]], "episode": e, "constant": constant, "out": str(out)}
            for e in lock["episodes"]]
    worker = importlib.import_module(Path(__file__).stem)  # cv_host_run runs this file in a fresh namespace
    with Pool(args.workers) as pool:
        done = pool.map(worker.run, jobs, chunksize=1)
    predictions, purposes = {}, set()
    for results, used in done:
        purposes |= set(used)
        for item in results:
            predictions[item["key"]] = {k: item[k] for k in ("path", "sha256", "kind", "hit7")}
    if purposes - {"ORACLE_TEACHER_ONLY", "QUERY_CAMERA_TEACHER_ONLY"}:
        raise PermissionError(f"unexpected read purposes {sorted(purposes)}")
    replay = None
    if args.cohort == "CAJU_COMPAT":
        replay = {"max_abs_diff_r7": 0.0, "max_abs_diff_cosine": 0.0, "near7_mismatch": [], "missing": []}
        for key, item in predictions.items():
            if item["kind"] != "query":
                continue
            sealed = v3["predictions"].get(key)
            if sealed is None:
                replay["missing"].append(key)
                continue
            if sha(sealed["path"]) != sealed["sha256"]:
                raise PermissionError(f"V3 geometry file changed: {key}")
            ours, theirs = np.load(item["path"]), np.load(sealed["path"])
            replay["max_abs_diff_r7"] = max(replay["max_abs_diff_r7"], float(np.max(np.abs(
                ours["REPROJ7_HARMONIC"] - theirs["REPROJ7_HARMONIC"]))))
            replay["max_abs_diff_cosine"] = max(replay["max_abs_diff_cosine"], float(np.max(np.abs(
                ours["cosine"] - theirs["cosine"]))))
            if not np.array_equal(ours["near7"].astype(bool), theirs["near7"].astype(bool)):
                replay["near7_mismatch"].append(key)
        replay["pass"] = (replay["max_abs_diff_r7"] <= REPLAY_TOLERANCE and replay["max_abs_diff_cosine"]
                          <= REPLAY_TOLERANCE and not replay["near7_mismatch"] and not replay["missing"])
    report = {"cohort": args.cohort, "episodes": len(jobs), "targets": len(predictions), "query_depth_read": False,
              "history_depth_read": False, "read_purposes": sorted(purposes), "reprojection_fill_constant_m": constant,
              "constant_source": str(V3_SEAL), "constant_source_sha256": sha(V3_SEAL),
              "frozen_sources": g.frozen_hashes(), "replay_vs_v3": replay, "predictions": predictions}
    (root / "geometry" / f"geometry_seal_{args.cohort}.json").write_text(json.dumps(report, indent=1))
    print("CAJU_GEOMETRY_SEALED", args.cohort, len(predictions), "targets; replay", replay)
    if replay is not None and not replay["pass"]:
        raise SystemExit("REPLAY_FAILED")


if __name__ == "__main__":
    main()
