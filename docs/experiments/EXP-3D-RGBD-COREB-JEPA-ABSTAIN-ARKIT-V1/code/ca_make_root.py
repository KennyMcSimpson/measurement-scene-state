"""EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1: the root, two evaluation sub-roots and the reused seals (no prediction is
made, no query depth is read). ds_eval.py reads ROOT/frame_role_lock.json, so the DEV cohorts get sub-roots holding byte
copies of their frame-role locks (eval_SYSTEM/ from FULL-SYSTEM-V1, eval_SYSTEM2/ from FULL-SYSTEM-V2) for the JN1 CACHE
evaluations of this experiment. Recorded with hashes, per cohort: the frame-role lock, the geometry seal and the reused
DepthSplat seals (cap 0.1):
  DEV      ARKIT_SYSTEM, ARKIT_SYSTEM2: ZS = the released CACHE + TTT_m1 run
  CONFIRM  ARKIT_SYSTEM3: ZS of V3 (CACHE + TTT_m1), JN1-JN3 of NF;  ARKIT_VAL: ZS of ENSEMBLES-ARKIT, JN1 of
           OBJECTIVE-TTT, JN2 / JN3 of NS
and the JN1 checkpoint of OBJECTIVE-TTT (checked against its SHA256SUMS).
Run from the repository root:  python3 ca_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

OUT = Path("outputs").resolve()
SYS1, SYS2 = OUT / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1", OUT / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2"
V3, ENS = OUT / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3", OUT / "EXP-3D-RGBD-ENSEMBLES-ARKIT-V1"
NF = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1"
NS = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1"
OBJ = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1"
COHORTS = {
    "ARKIT_SYSTEM": ("DEV", SYS1, "eval_SYSTEM", {"ZS": (SYS1, "ZS", True)}),
    "ARKIT_SYSTEM2": ("DEV", SYS2, "eval_SYSTEM2", {"ZS": (SYS2, "ZS", True)}),
    "ARKIT_SYSTEM3": ("CONFIRM", V3, None, {"ZS": (V3, "ZS", True), **{f"JN{i}": (NF, f"JN{i}", False) for i in (1, 2, 3)}}),
    "ARKIT_VAL": ("CONFIRM", ENS, None, {"ZS": (ENS, "ZS", True), "JN1": (OBJ, "JN", True), "JN2": (NS, "JN2", False),
                                         "JN3": (NS, "JN3", False)}),
}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parts(directory, cohort, keys, ttt):
    files = sorted(Path(directory).glob("ds_seal_*of*.json"))
    covered = set()
    for part in files:
        seal = json.loads(part.read_text())
        if (seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != cohort
                or seal["scale_max"] != 0.1 or "CACHE" not in seal["arms"]):
            raise PermissionError(f"{part}: not a complete CACHE run at cap 0.1")
        if ttt and ("TTT" not in seal["arms"] or seal["ttt_mult"] != [1.0] or seal["ttt_steps"] != 100):
            raise PermissionError(f"{part}: not a TTT_m1 run of 100 steps")
        covered |= set(seal["predictions"])
    if covered != keys:
        raise ValueError(f"{directory}: the seals do not cover the frame-role lock")
    return [{"path": str(p), "sha256": sha(p)} for p in files]


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
reused = {"cohorts": {}, "checkpoints": {}}
for cohort, (role, source, sub, seals) in COHORTS.items():
    lock_path = source / "frame_role_lock.json"
    if sub:
        (root / sub).mkdir()
        shutil.copyfile(lock_path, root / sub / "frame_role_lock.json")
    lock = json.loads(lock_path.read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
    keys = {f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]}
    geometry = source / f"geometry/geometry_seal_{cohort}.json"
    seal = json.loads(geometry.read_text())
    if seal["query_depth_read"] is not False or not seal["replay_vs_arms_seal"]["pass"] or set(seal["predictions"]) != keys:
        raise PermissionError(f"{cohort}: geometry seal incomplete, not certified or its replay failed")
    reused["cohorts"][cohort] = {"role": role, "sub_root": sub,
                                 "frame_role_lock": {"path": str(lock_path), "sha256": sha(lock_path)},
                                 "geometry": {"path": str(geometry), "sha256": sha(geometry)},
                                 "seals": {name: parts(experiment / f"ds/{cohort}/{folder}", cohort, keys, ttt)
                                           for name, (experiment, folder, ttt) in seals.items()}}
path = OBJ / "checkpoints/JN1/depthsplat_jepa_final.pth"
listed = (OBJ / "checkpoints/JN1/SHA256SUMS").read_text().split()[0]
if sha(path) != listed:
    raise PermissionError("JN1: the checkpoint differs from its SHA256SUMS")
reused["checkpoints"]["JN1"] = {"path": str(path), "sha256": listed}
(root / "reused_seals.json").write_text(json.dumps(reused, indent=1) + "\n")
print("CA_ROOT", root, {c: (v["role"], sorted(v["seals"])) for c, v in reused["cohorts"].items()})
