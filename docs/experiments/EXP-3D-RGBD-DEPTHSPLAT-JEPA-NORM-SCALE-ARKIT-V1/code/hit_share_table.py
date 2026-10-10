"""Development diagnostic (not an experiment): for every ARKit record of a manifest, every context role (A, B) and
every non-context frame, the share of that frame's pixels hit by the 1-pixel z-buffer reprojection of the role's
three context frames (frozen measurement_planning.geometry functions). Uses context depth and the target camera only;
no target depth is read. Output: one JSON per manifest with {scene_id: {role: {frame_id: hit_share}}}.
Run through cv_host_run.py from the repository root (CUDA hidden):
  python hit_share_table.py MANIFEST SPLIT OUT.json [--workers 8]
"""
import argparse
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


def run(job):
    torch.set_num_threads(1)
    record = job["record"]
    out = {}
    for role, key in (("A", "context_a"), ("B", "context_b")):
        context = list(record["roles"][key])
        targets = [f["frame_id"] for f in record["frames"] if f["frame_id"] not in context]
        log = []
        teacher = TeacherAccess(record, targets, IMAGE_SIZE, REPO, log)
        points = g.frame_points(teacher.frames(context))
        shares = {}
        for fid in targets:
            camera = teacher.query_camera(fid)
            hit = np.isfinite(g.union([g.reproject(points[f], camera) for f in context]))
            shares[str(fid)] = float(hit.mean())
        if {e["purpose"] for e in log} - {"ORACLE_TEACHER_ONLY", "QUERY_CAMERA_TEACHER_ONLY"}:
            raise PermissionError("unexpected read purpose")
        out[role] = shares
    return record["scene_id"], out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("split")
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    records = [r for r in json.loads(args.manifest.read_text())["scenes"] if r["split"] == args.split]
    worker = importlib.import_module(Path(__file__).stem)
    with Pool(args.workers) as pool:
        table = dict(pool.map(worker.run, [{"record": r} for r in records], chunksize=1))
    args.out.write_text(json.dumps(table, indent=0))
    shares = np.array([v for roles in table.values() for frames in roles.values() for v in frames.values()])
    print("HIT_SHARES", args.split, len(records), "records", shares.size, "pairs; zero", round(float(np.mean(shares == 0)), 3),
          "in [0.05, 0.95]", round(float(np.mean((shares >= 0.05) & (shares <= 0.95))), 3),
          "median", round(float(np.median(shares)), 3))
    by_frame = {}
    for roles in table.values():
        for frames in roles.values():
            for fid, v in frames.items():
                by_frame.setdefault(int(fid), []).append(v)
    print("  zero-hit share by frame:", {f: round(float(np.mean(np.array(v) == 0)), 2) for f, v in sorted(by_frame.items())})
    print("  median hit share by frame:", {f: round(float(np.median(v)), 2) for f, v in sorted(by_frame.items())})


if __name__ == "__main__":
    main()
