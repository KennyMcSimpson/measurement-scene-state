"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1: the evaluation root. Copies the frame-role lock of
EXP-3D-RGBD-ENSEMBLES-ARKIT-V1 byte for byte (ARKIT_VAL: context = role frames, stream = the Core B rule, queries =
primary_query) and records the reused seals with their hashes: the 7-frame geometry seal and the released-weights ZS
seals (scale cap 0.1: CACHE, TTT_m1) of ENSEMBLES-ARKIT, and the JA and ZS3 seals (cap 3.0: CACHE, TTT_m1) of
EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1. No prediction is made and no query depth is read.
Run from the repository root:  python3 jc_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

SOURCE = Path("outputs/EXP-3D-RGBD-ENSEMBLES-ARKIT-V1").resolve()
JBT = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1").resolve()
COHORT = "ARKIT_VAL"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_parts(directory, scale_max, keys):
    parts = sorted(Path(directory).glob("ds_seal_*of*.json"))
    covered = set()
    for part in parts:
        seal = json.loads(part.read_text())
        if (seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT
                or seal["scale_max"] != scale_max or seal["arms"] != ["CACHE", "TTT"] or seal["ttt_mult"] != [1.0]
                or seal["ttt_steps"] != 100):
            raise PermissionError(f"{part}: not a complete CACHE / TTT_m1 run at cap {scale_max}")
        covered |= set(seal["predictions"])
    if covered != keys:
        raise ValueError(f"{directory}: the seals do not cover the frame-role lock")
    return [{"path": str(p), "sha256": sha(p)} for p in parts]


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
shutil.copyfile(SOURCE / "frame_role_lock.json", root / "frame_role_lock.json")
if sha(JBT / "frame_role_lock.json") != sha(root / "frame_role_lock.json"):
    raise PermissionError("JBT used a different frame-role lock")
lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
if sha(lock["manifest"]) != lock["manifest_sha256"]:
    raise PermissionError("manifest changed since the frame-role lock")
keys = {f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]}
geometry = SOURCE / f"geometry/geometry_seal_{COHORT}.json"
seal = json.loads(geometry.read_text())
if seal["query_depth_read"] is not False or not seal["replay_vs_arms_seal"]["pass"] or set(seal["predictions"]) != keys:
    raise PermissionError("geometry seal incomplete, not certified or its replay failed")
reused = {"source_experiment": str(SOURCE), "jbt_experiment": str(JBT),
          "frame_role_lock_sha256": sha(SOURCE / "frame_role_lock.json"),
          "geometry_seal": {"path": str(geometry), "sha256": sha(geometry)},
          "seals": {"ZS": seal_parts(SOURCE / f"ds/{COHORT}/ZS", 0.1, keys),
                    "JA": seal_parts(JBT / f"ds/{COHORT}/JA", 3.0, keys),
                    "ZS3": seal_parts(JBT / f"ds/{COHORT}/ZS3", 3.0, keys)}}
(root / "reused_seals.json").write_text(json.dumps(reused, indent=1) + "\n")
print("JC_ROOT", root, len(lock["episodes"]), "episodes", len(keys), "queries; reused",
      {k: len(v) for k, v in reused["seals"].items()})
