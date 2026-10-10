"""EXP-3D-RGBD-COREB-WEIGHT-ARKIT-V1: the root and its reused seals (no prediction is made, no query depth is read). Per
cohort, with hashes: the frame-role lock, the geometry seal and the released-weights CACHE + TTT_m1 seals (cap 0.1):
  DEV      ARKIT_SYSTEM (FULL-SYSTEM-V1), ARKIT_SYSTEM2 (FULL-SYSTEM-V2)
  CONFIRM  ARKIT_SYSTEM3 (FULL-SYSTEM-V3), ARKIT_VAL (ENSEMBLES-ARKIT)
Run from the repository root:  python3 cw_make_root.py ROOT
"""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path("outputs").resolve()
COHORTS = {"ARKIT_SYSTEM": ("DEV", OUT / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1"),
           "ARKIT_SYSTEM2": ("DEV", OUT / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2"),
           "ARKIT_SYSTEM3": ("CONFIRM", OUT / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3"),
           "ARKIT_VAL": ("CONFIRM", OUT / "EXP-3D-RGBD-ENSEMBLES-ARKIT-V1")}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
reused = {}
for cohort, (role, source) in COHORTS.items():
    lock_path = source / "frame_role_lock.json"
    lock = json.loads(lock_path.read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
    keys = {f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]}
    geometry = source / f"geometry/geometry_seal_{cohort}.json"
    seal = json.loads(geometry.read_text())
    if seal["query_depth_read"] is not False or not seal["replay_vs_arms_seal"]["pass"] or set(seal["predictions"]) != keys:
        raise PermissionError(f"{cohort}: geometry seal incomplete, not certified or its replay failed")
    files, covered = sorted((source / f"ds/{cohort}/ZS").glob("ds_seal_*of*.json")), set()
    for part in files:
        s = json.loads(part.read_text())
        if (s["query_depth_read"] is not False or s["limit"] or s["cohort"] != cohort or s["scale_max"] != 0.1
                or "TTT" not in s["arms"] or s["ttt_mult"] != [1.0] or s["ttt_steps"] != 100):
            raise PermissionError(f"{part}: not a complete TTT_m1 run at cap 0.1")
        covered |= set(s["predictions"])
    if covered != keys:
        raise ValueError(f"{cohort}: the ZS seals do not cover the frame-role lock")
    reused[cohort] = {"role": role, "frame_role_lock": {"path": str(lock_path), "sha256": sha(lock_path)},
                      "geometry": {"path": str(geometry), "sha256": sha(geometry)},
                      "zs": [{"path": str(p), "sha256": sha(p)} for p in files]}
(root / "reused_seals.json").write_text(json.dumps(reused, indent=1) + "\n")
print("CW_ROOT", root, {c: (v["role"], len(v["zs"])) for c, v in reused.items()})
