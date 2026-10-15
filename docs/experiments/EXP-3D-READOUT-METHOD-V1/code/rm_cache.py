"""EXP-3D-READOUT-METHOD-V1: one input cache per cohort, from sealed predictions only (no ground truth is read).

Per query (128 x 160): R7 (REPROJ7_HARMONIC), near7, cosine; the released-weights feed-forward render (CACHE: depth_z,
alpha) and the 100-step TTT render (TTT: depth_z, alpha); CAJU also keeps its 10-step snapshot (SHORT). Every sealed
file is checked against its hash. Writes cache/<cohort>.npz and cache/<cohort>.json (keys, places, episodes, hit7,
source seals).
Run from the repository root (CUDA hidden, PYTHONPATH=src):  python rm_cache.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs").resolve()
for _p in ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code", "EXP-3D-CAJU-HEADROOM-V1/code", "EXP-3D-CAJU3D-ENGINE-V1/code"):
    sys.path.insert(0, str(OUT / _p))
import engine_score as S  # noqa: E402

KF = OUT / "EXP-3D-CAJU3D-KFUSE-V1"
CAJU = OUT / "EXP-3D-CAJU-HEADROOM-V1"
S4 = OUT / "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(root, cohort, rows, arrays, sources):
    np.savez(root / "cache" / f"{cohort}.npz", **{k: np.stack(v) for k, v in arrays.items()})
    (root / "cache" / f"{cohort}.json").write_text(json.dumps({"cohort": cohort, "queries": rows, "sources": sources},
                                                              indent=1) + "\n")
    print(cohort, len(rows), "queries", flush=True)


def kf_cohort(root, cohort):
    lock = json.loads((KF / "frame_role_lock.json").read_text())["cohorts"][cohort]
    reused = json.loads((KF / "reused_seals.json").read_text())[cohort]
    if sha(reused["geometry"]["path"]) != reused["geometry"]["sha256"]:
        raise PermissionError(f"{cohort}: geometry seal changed")
    geo = json.loads(Path(reused["geometry"]["path"]).read_text())["predictions"]
    zs = S.reused_predictions(reused["ZS"])
    rows, arrays = [], {k: [] for k in ("r7", "near7", "cosine", "ttt_z", "ttt_a", "ff_z", "ff_a")}
    for e in lock["episodes"]:
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            g, t = S.load_checked(geo[key]), S.load_checked(zs[key])
            for name, value in (("r7", g["REPROJ7_HARMONIC"]), ("near7", g["near7"]), ("cosine", g["cosine"]),
                                ("ttt_z", t["TTT_m1_depth_z"]), ("ttt_a", t["TTT_m1_alpha"]),
                                ("ff_z", t["CACHE_depth_z"]), ("ff_a", t["CACHE_alpha"])):
                arrays[name].append(value.astype(np.float32) if name != "near7" else value.astype(bool))
            rows.append({"key": key, "place": e["place"], "episode": f"{e['scene_id']}/{e['role']}", "hit7": geo[key]["hit7"],
                         "scene_id": e["scene_id"], "query": qid})
    write(root, cohort, rows, arrays, {"frame_role_lock": str(KF / "frame_role_lock.json"),
                                       "reused_seals": str(KF / "reused_seals.json")})


def caju_cohorts(root):
    replay = {}
    for part in sorted((CAJU / "replay").glob("*/replay_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["ttt_steps"] != 100 or not seal["checkpoint_sha256"].startswith("f8ddd845"):
            raise PermissionError(f"{part}: not the released-weights 100-step replay")
        replay.update({k: v for k, v in seal["predictions"].items() if v["kind"] == "query"})
    locks = json.loads((CAJU / "frame_role_lock.json").read_text())["cohorts"]
    rows, arrays = [], {k: [] for k in ("r7", "near7", "cosine", "ttt_z", "ttt_a", "ff_z", "ff_a", "short_z", "short_a")}
    for cohort in ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL"):
        geo = json.loads((CAJU / "geometry" / f"geometry_seal_{cohort}.json").read_text())["predictions"]
        for e in locks[cohort]["episodes"]:
            for qid in e["queries"]:
                key = f"{e['scene_id']}/{e['role']}/{qid}"
                g, t = S.load_checked(geo[key]), S.load_checked(replay[key])
                for name, value in (("r7", g["REPROJ7_HARMONIC"]), ("near7", g["near7"]), ("cosine", g["cosine"]),
                                    ("ttt_z", t["FULL_depth_z"]), ("ttt_a", t["FULL_alpha"]), ("ff_z", t["OFF_depth_z"]),
                                    ("ff_a", t["OFF_alpha"]), ("short_z", t["SHORT_depth_z"]),
                                    ("short_a", t["SHORT_alpha"])):
                    arrays[name].append(value.astype(np.float32) if name != "near7" else value.astype(bool))
                rows.append({"key": key, "place": e["place"], "episode": f"{e['scene_id']}/{e['role']}",
                             "hit7": geo[key]["hit7"], "scene_id": e["scene_id"], "query": qid, "source_cohort": cohort})
    write(root, "CAJU", rows, arrays, {"frame_role_lock": str(CAJU / "frame_role_lock.json"),
                                       "replay": str(CAJU / "replay"), "geometry": str(CAJU / "geometry")})


def system4(root):
    lock = json.loads((S4 / "frame_role_lock.json").read_text())["cohorts"]["ARKIT_SYSTEM4"]
    recs = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    gseal = json.loads((S4 / "geometry/geometry_seal_ARKIT_SYSTEM4.json").read_text())
    geo = gseal["predictions"]
    ttt = {}
    for part in sorted((S4 / "ds/ARKIT_SYSTEM4/ZS").glob("ds_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["query_depth_read"] is not False or seal["scale_max"] != 0.1 or seal["limit"]:
            raise PermissionError(f"{part}: not the protocol's run")
        ttt.update(seal["predictions"])
    rows, arrays = [], {k: [] for k in ("r7", "near7", "cosine", "ttt_z", "ttt_a", "ff_z", "ff_a")}
    for e in lock["episodes"]:
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            g, t = S.load_checked(geo[key]), S.load_checked(ttt[key])
            for name, value in (("r7", g["REPROJ7_HARMONIC"]), ("near7", g["near7"]), ("cosine", g["cosine"]),
                                ("ttt_z", t["TTT_m1_depth_z"]), ("ttt_a", t["TTT_m1_alpha"]),
                                ("ff_z", t["CACHE_depth_z"]), ("ff_a", t["CACHE_alpha"])):
                arrays[name].append(value.astype(np.float32) if name != "near7" else value.astype(bool))
            rows.append({"key": key, "place": recs[e["scene_id"]]["physical_scene_id"],
                         "episode": f"{e['scene_id']}/{e['role']}", "hit7": geo[key]["hit7"], "scene_id": e["scene_id"],
                         "query": qid})
    write(root, "ARKIT_SYSTEM4", rows, arrays, {"frame_role_lock": str(S4 / "frame_role_lock.json"),
                                                "geometry": str(S4 / "geometry"), "ds": str(S4 / "ds")})


def main():
    root = Path(sys.argv[1]).resolve()
    (root / "cache").mkdir(parents=True, exist_ok=True)
    for cohort in ("FIT_SYSTEM", "FIT_SYSTEM2", "CONF_SYSTEM3", "CONF_VAL"):
        kf_cohort(root, cohort)
    caju_cohorts(root)
    system4(root)
    print("RM_CACHE_DONE")


if __name__ == "__main__":
    main()
