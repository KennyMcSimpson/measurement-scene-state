"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1: the evaluation root. Copies the frame-role lock of
EXP-3D-RGBD-ENSEMBLES-ARKIT-V1 byte for byte (ARKIT_VAL: context = role frames, stream = the Core B rule, queries =
primary_query) and records the reused seals with their hashes: the 7-frame geometry seal (REPROJ7_HARMONIC, NEAR7,
cosine) and the released-weights DepthSplat seals (ZS, scale cap 0.1: CACHE, TTT_m1). No prediction is made and no
query depth is read. Run from the repository root:  python3 jbt_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

SOURCE = Path("outputs/EXP-3D-RGBD-ENSEMBLES-ARKIT-V1").resolve()
COHORT = "ARKIT_VAL"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
shutil.copyfile(SOURCE / "frame_role_lock.json", root / "frame_role_lock.json")
lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
if sha(lock["manifest"]) != lock["manifest_sha256"]:
    raise PermissionError("manifest changed since the frame-role lock")
if any(len(e["stream"]) != 4 or len(e["context"]) != 3 for e in lock["episodes"]):
    raise ValueError("every episode needs three context and four stream frames")
keys = {f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]}
geometry = SOURCE / f"geometry/geometry_seal_{COHORT}.json"
seal = json.loads(geometry.read_text())
if seal["query_depth_read"] is not False or not seal["replay_vs_arms_seal"]["pass"] or set(seal["predictions"]) != keys:
    raise PermissionError("geometry seal incomplete, not certified or its replay failed")
parts = sorted((SOURCE / f"ds/{COHORT}/ZS").glob("ds_seal_*of*.json"))
covered = set()
for part in parts:
    zs = json.loads(part.read_text())
    if (zs["query_depth_read"] is not False or zs["limit"] or zs["cohort"] != COHORT or zs["scale_max"] != 0.1
            or zs["arms"] != ["CACHE", "TTT"] or zs["ttt_mult"] != [1.0] or zs["ttt_steps"] != 100):
        raise PermissionError(f"{part}: not the released-weights CACHE / TTT_m1 run")
    covered |= set(zs["predictions"])
if covered != keys:
    raise ValueError("the ZS seals do not cover the frame-role lock")
reused = {"source_experiment": str(SOURCE), "frame_role_lock_sha256": sha(SOURCE / "frame_role_lock.json"),
          "geometry_seal": {"path": str(geometry), "sha256": sha(geometry)},
          "zs_seals": [{"path": str(p), "sha256": sha(p)} for p in parts]}
(root / "reused_seals.json").write_text(json.dumps(reused, indent=1) + "\n")
print("JBT_ROOT", root, len(lock["episodes"]), "episodes", len(keys), "queries; ZS parts", len(parts))
