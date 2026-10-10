"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1: the root (no prediction is made, no query depth is read). The
sub-root eval_OVERLAP/ holds a byte copy of the OVERLAP-V1 frame-role lock (ARKIT160_OVERLAP; ds_eval.py reads
ROOT/frame_role_lock.json). Recorded with hashes: that lock, the OVERLAP-V1 arms seal (REPROJ_HARMONIC, NEAR, G1) and the
JO1-JO3 CTX seals (cap 3.0) of OVERLAP-V1.
Run from the repository root:  python3 os_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

OVERLAP = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1").resolve()
COHORT = "ARKIT160_OVERLAP"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
(root / "eval_OVERLAP").mkdir()
shutil.copyfile(OVERLAP / "frame_role_lock.json", root / "eval_OVERLAP/frame_role_lock.json")
lock = json.loads((root / "eval_OVERLAP/frame_role_lock.json").read_text())["cohorts"][COHORT]
if sha(lock["manifest"]) != lock["manifest_sha256"]:
    raise PermissionError("manifest changed since the frame-role lock")
keys = {f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]}
arms = OVERLAP / f"raw/eval/{COHORT}/prediction_seal.json"
seal = json.loads(arms.read_text())
if seal["query_depth_read"] is not False or any(f"{COHORT}/{k}" not in seal["predictions"] for k in keys):
    raise PermissionError("arms seal incomplete or not certified")
jo = {}
for name in ("JO1", "JO2", "JO3"):
    files, covered = sorted((OVERLAP / f"ds/{COHORT}/{name}").glob("ds_seal_*of*.json")), set()
    for part in files:
        s = json.loads(part.read_text())
        if s["query_depth_read"] is not False or s["limit"] or s["cohort"] != COHORT or s["scale_max"] != 3.0 or "CTX" not in s["arms"]:
            raise PermissionError(f"{part}: not a complete CTX run at cap 3.0")
        covered |= set(s["predictions"])
    if covered != keys:
        raise ValueError(f"{name}: the seals do not cover the frame-role lock")
    jo[name] = [{"path": str(p), "sha256": sha(p)} for p in files]
reused = {"overlap_experiment": str(OVERLAP),
          "frame_role_lock": {"source": str(OVERLAP / "frame_role_lock.json"),
                              "sha256": sha(root / "eval_OVERLAP/frame_role_lock.json")},
          "arms_seal": {"path": str(arms), "sha256": sha(arms)}, "jo_seals": jo}
(root / "reused_seals.json").write_text(json.dumps(reused, indent=1) + "\n")
print("OS_ROOT", root, len(lock["episodes"]), "episodes", len(keys), "queries")
