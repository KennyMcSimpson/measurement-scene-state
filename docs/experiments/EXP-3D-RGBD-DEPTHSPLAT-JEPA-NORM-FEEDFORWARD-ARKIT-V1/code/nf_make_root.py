"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1: the evaluation root. Copies the frame-role lock of
EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3 byte for byte (ARKIT_SYSTEM3: context = role frames, stream = the Core B rule,
queries = primary_query) and records, with hashes, the reused 7-frame geometry seal of V3 and the JN1 / JK1 checkpoints
of EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1 (checked against their SHA256SUMS). Refuses once V3 has scored,
so no ARKIT_SYSTEM3 query depth has been read when the design is fixed. No prediction is made and no query depth is read.
Run from the repository root:  python3 nf_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

V3 = Path("outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3").resolve()
OBJ = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1").resolve()
COHORT = "ARKIT_SYSTEM3"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
if (V3 / "scores").exists():
    raise PermissionError("V3 has scored: ARKIT_SYSTEM3 query depth was read")
root.mkdir(parents=True, exist_ok=False)
shutil.copyfile(V3 / "frame_role_lock.json", root / "frame_role_lock.json")
lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
if sha(lock["manifest"]) != lock["manifest_sha256"]:
    raise PermissionError("manifest changed since the frame-role lock")
keys = {f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]}
geometry = V3 / f"geometry/geometry_seal_{COHORT}.json"
seal = json.loads(geometry.read_text())
if seal["query_depth_read"] is not False or not seal["replay_vs_arms_seal"]["pass"] or set(seal["predictions"]) != keys:
    raise PermissionError("geometry seal incomplete, not certified or its replay failed")
checkpoints = {}
for name in ("JN1", "JK1"):
    path = OBJ / f"checkpoints/{name}/depthsplat_jepa_final.pth"
    listed = (OBJ / f"checkpoints/{name}/SHA256SUMS").read_text().split()[0]
    if sha(path) != listed:
        raise PermissionError(f"{name}: the checkpoint differs from its SHA256SUMS")
    checkpoints[name] = {"path": str(path), "sha256": listed}
reused = {"v3_experiment": str(V3), "objective_experiment": str(OBJ),
          "frame_role_lock_sha256": sha(V3 / "frame_role_lock.json"),
          "geometry_seal": {"path": str(geometry), "sha256": sha(geometry)}, "checkpoints": checkpoints}
(root / "reused_seals.json").write_text(json.dumps(reused, indent=1) + "\n")
print("NF_ROOT", root, len(lock["episodes"]), "episodes", len(keys), "queries; checkpoints", sorted(checkpoints))
