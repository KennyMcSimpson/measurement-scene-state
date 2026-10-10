"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1: the evaluation sub-roots and the reused seals. ds_eval.py reads
ROOT/frame_role_lock.json, so each cohort gets a sub-root holding a byte copy of an earlier frame-role lock:
  eval_SYSTEM3/  from EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3 (ARKIT_SYSTEM3, 160 scans; NF used the same lock)
  eval_VAL/      from EXP-3D-RGBD-ENSEMBLES-ARKIT-V1 (ARKIT_VAL, 80 scans)
It records, with hashes: the geometry seals; the reused DepthSplat seals (cap 0.1), ARKIT_SYSTEM3: ZS (CACHE) and
JN1-JN3 of NF, ZS_TTT (CACHE + TTT_m1) of V3; ARKIT_VAL: ZS and ZS_TTT (one CACHE + TTT_m1 run) of ENSEMBLES-ARKIT, JN1
of OBJECTIVE-TTT; and the NF checkpoints JN2 / JN3 (checked against their SHA256SUMS), evaluated on ARKIT_VAL here.
No prediction is made and no query depth is read.
Run from the repository root:  python3 ns_make_root.py ROOT
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

OUT = Path("outputs").resolve()
V3 = OUT / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3"
NF = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1"
ENS = OUT / "EXP-3D-RGBD-ENSEMBLES-ARKIT-V1"
OBJ = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1"
COHORTS = {"ARKIT_SYSTEM3": ("eval_SYSTEM3", V3), "ARKIT_VAL": ("eval_VAL", ENS)}
REUSED = {"ARKIT_SYSTEM3": {"ZS": (NF, "ZS", False), "JN1": (NF, "JN1", False), "JN2": (NF, "JN2", False),
                            "JN3": (NF, "JN3", False), "ZS_TTT": (V3, "ZS", True)},
          "ARKIT_VAL": {"ZS": (ENS, "ZS", True), "ZS_TTT": (ENS, "ZS", True), "JN1": (OBJ, "JN", True)}}


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
reused = {"sources": {k: str(v) for k, v in (("V3", V3), ("NF", NF), ("ENSEMBLES-ARKIT", ENS), ("OBJECTIVE-TTT", OBJ))},
          "frame_role_locks": {}, "geometry": {}, "seals": {}, "checkpoints": {}}
for cohort, (sub, source) in COHORTS.items():
    (root / sub).mkdir()
    shutil.copyfile(source / "frame_role_lock.json", root / sub / "frame_role_lock.json")
    reused["frame_role_locks"][cohort] = {"source": str(source / "frame_role_lock.json"),
                                          "sha256": sha(root / sub / "frame_role_lock.json")}
    lock = json.loads((root / sub / "frame_role_lock.json").read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
    keys = {f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]}
    geometry = source / f"geometry/geometry_seal_{cohort}.json"
    seal = json.loads(geometry.read_text())
    if seal["query_depth_read"] is not False or not seal["replay_vs_arms_seal"]["pass"] or set(seal["predictions"]) != keys:
        raise PermissionError(f"{cohort}: geometry seal incomplete, not certified or its replay failed")
    reused["geometry"][cohort] = {"path": str(geometry), "sha256": sha(geometry)}
    reused["seals"][cohort] = {name: parts(experiment / f"ds/{cohort}/{folder}", cohort, keys, ttt)
                               for name, (experiment, folder, ttt) in REUSED[cohort].items()}
if sha(root / "eval_SYSTEM3/frame_role_lock.json") != sha(NF / "frame_role_lock.json"):
    raise PermissionError("NF used a different ARKIT_SYSTEM3 lock")
for name in ("JN2", "JN3"):
    path = NF / f"checkpoints/{name}/depthsplat_jepa_final.pth"
    listed = (NF / f"checkpoints/{name}/SHA256SUMS").read_text().split()[0]
    if sha(path) != listed:
        raise PermissionError(f"{name}: the checkpoint differs from its SHA256SUMS")
    reused["checkpoints"][name] = {"path": str(path), "sha256": listed}
(root / "reused_seals.json").write_text(json.dumps(reused, indent=1) + "\n")
print("NS_ROOT", root, {c: len(v) for c, v in reused["seals"].items()}, sorted(reused["checkpoints"]))
