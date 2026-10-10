"""DESCRIPTIVE, post hoc (after EXP-3D-RGBD-TTT-ENSEMBLE-V1): the TTT ensemble on Replica (8 scenes, 16 episodes, the
Core B V28 streams). Too few scenes for the V2 gate; reports scene means and how many scenes improve.
Stages (run through cv_host_run.py from the repository root, CUDA hidden):
  python replica_tttens.py ROOT make     root: Replica manifest copy + frame-role lock (streams = V28's, rule-checked)
  (then V1's geometry_seal.py ROOT REPLICA <Replica arms seal>, and V1's ds_eval.py ZS CACHE TTT on REPLICA)
  python replica_tttens.py ROOT score    R7, CACHE, TTT, sqrt(R7 x TTT), sqrt(R7 x CACHE) on ALL / NEAR7 / FAR7
"""
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path.cwd()
SOURCE = REPO / "outputs/EXP-3D-RGBD-REPLICA-DATA-V1/manifest_replica.json"
V28 = REPO / "outputs/EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28/stream_roles.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    assert sha(item["path"]) == item["sha256"], item["path"]
    return dict(np.load(item["path"]))


root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
if stage == "make":
    from mcss.mechanism_pilot.rgbd_stream_write import stream_frame_ids

    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE, root / "manifest_replica.json")
    records = json.loads((root / "manifest_replica.json").read_text())["scenes"]
    sealed = json.loads(V28.read_text())["episodes"]
    episodes = []
    for r in records:
        stream = stream_frame_ids(r, "A")
        for role, key in (("A", "context_a"), ("B", "context_b")):
            assert sealed[f"{r['scene_id']}/{role}"] == list(stream), r["scene_id"]
            episodes.append({"scene_id": r["scene_id"], "role": role, "context": list(r["roles"][key]),
                             "stream": list(stream), "queries": list(r["roles"]["primary_query"])})
    lock = {"rule": "context = role frames; stream = Core B frame-id rule (equal to V28's); queries = primary_query",
            "cohorts": {"REPLICA": {"manifest": str(root / "manifest_replica.json"),
                                    "manifest_sha256": sha(root / "manifest_replica.json"), "episodes": episodes}}}
    (root / "frame_role_lock.json").write_text(json.dumps(lock, indent=1))
    print("REPLICA_TTT_ROOT", root, len(episodes), "episodes")
else:
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"]["REPLICA"]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    geometry = json.loads((root / "geometry/geometry_seal_REPLICA.json").read_text())
    assert geometry["replay_vs_arms_seal"]["pass"]
    zs = {}
    for part in sorted((root / "ds/REPLICA/ZS").glob("ds_seal_*of*.json")):
        seal = json.loads(part.read_text())
        assert seal["query_depth_read"] is False and not seal["limit"]
        zs.update(seal["predictions"])
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo, arrays = load(geometry["predictions"][key]), load(zs[key])
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            valid = np.isfinite(gt) & (gt > 0)
            near7 = geo["near7"].astype(bool)
            r7, cosine = geo["REPROJ7_HARMONIC"], geo["cosine"]
            state = {}
            for arm in ("CACHE", "TTT_m1"):
                alpha = arrays[f"{arm}_alpha"].astype(np.float64)
                ray = arrays[f"{arm}_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
                state[arm] = np.where(alpha >= 0.5, ray, r7)
            gm = lambda a, b: np.sqrt(np.maximum(a, 1e-6) * np.maximum(b, 1e-6))  # noqa: E731
            readouts = {"R7": r7, "CACHE": state["CACHE"], "TTT": state["TTT_m1"],
                        "R7_TTT_GEO": gm(r7, state["TTT_m1"]), "R7_CACHE_GEO": gm(r7, state["CACHE"])}
            for name, p in readouts.items():
                for region, mask in (("ALL", valid), ("NEAR7", valid & near7), ("FAR7", valid & ~near7)):
                    if mask.any():
                        rows[f"{name}|{region}"][e["scene_id"]][e["role"]].append(
                            float(np.mean(np.abs(p[mask] - gt[mask]) / gt[mask])))
    table = {m: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for m, sc in rows.items()}
    out = {"note": "descriptive, post hoc; 8 Replica scenes; no gate",
           "means": {k: float(np.mean(list(v.values()))) for k, v in sorted(table.items())}, "scenes_better": {}}
    for left, right in (("R7", "TTT"), ("R7", "R7_TTT_GEO"), ("CACHE", "TTT"), ("R7_CACHE_GEO", "R7_TTT_GEO")):
        a, b = table[f"{left}|ALL"], table[f"{right}|ALL"]
        out["scenes_better"][f"{right} vs {left} (ALL)"] = [sum(a[s] > b[s] for s in a), len(a)]
    out["per_scene"] = table
    (root / "replica_tttens_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_scene"}, indent=1))
