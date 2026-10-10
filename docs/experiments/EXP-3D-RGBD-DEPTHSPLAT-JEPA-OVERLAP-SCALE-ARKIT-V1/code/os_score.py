"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1: scoring and the preregistered analysis on the VAL80 part of
ARKIT160_OVERLAP (source_cohort ARKIT_VAL; queries = frames 6 and 7, or the next free frame where a record uses one).

score: every prediction must be sealed and unchanged before query depth is read: the reused OVERLAP-V1 arms seal
(REPROJ_HARMONIC = H3, NEAR, G1 depth and opacity, three seeds) and JO1-JO3 CTX seals (hashes in reused_seals.json), and
this experiment's JOL1-JOL3 CTX seals (cap 3.0, each naming its trained checkpoint's hash). Per query (as OVERLAP-V1):
  JO, JOL  where at least two of the three seeds cover the pixel (alpha >= 0.5): the mean ray depth (depth_z / alpha /
           cos) of the covering seeds; elsewhere H3
  G1N      per G1 seed depth / opacity where opacity >= 0.5, else H3, mean over seeds
Regions: FARH = FAR (valid pixels outside the 8-px NEAR of the context reprojection) of the queries with at least one
context hit; FAR of every query; NEAR. Only VAL80 scenes are scored (the CONFIRM80 scans are JOL training data). Scene
value = mean over roles of the mean over queries (empty regions skipped).
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python os_score.py ROOT score | analyze
"""
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import contrast, write  # noqa: E402

COHORT, SOURCE, ALPHA_MIN = "ARKIT160_OVERLAP", "ARKIT_VAL", 0.5
BUILDERS = {"JO": ("JO1", "JO2", "JO3"), "JOL": ("JOL1", "JOL2", "JOL3")}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def ds_seal(files, keys, checkpoint=None):
    preds = {}
    for part in files:
        seal = json.loads(Path(part).read_text())
        if (seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT or seal["scale_max"] != 3.0
                or "CTX" not in seal["arms"]):
            raise PermissionError(f"{part}: not a complete CTX run at cap 3.0")
        if checkpoint is not None and seal["checkpoint_sha256"] != checkpoint:
            raise PermissionError(f"{part}: unexpected checkpoint")
        preds.update(seal["predictions"])
    if set(preds) != set(keys):
        raise ValueError(f"{files}: sealed queries differ from the frame-role lock")
    return preds


def ray_cosine(record, qid):
    from mcss.measurement_planning import geometry as g
    from mcss.measurement_planning.information import TeacherAccess

    teacher = TeacherAccess(record, [qid], (128, 160), Path.cwd(), [])
    return np.asarray(g.ray_cosine(teacher.query_camera(qid)), dtype=np.float64)


def pooled(arrays, cosine, fallback):
    covers = np.array([a["CTX_alpha"].astype(np.float64) >= ALPHA_MIN for a in arrays])
    planar = np.array([a["CTX_depth_z"].astype(np.float64) / np.maximum(a["CTX_alpha"].astype(np.float64), 1e-12)
                       for a in arrays])
    count = covers.sum(axis=0)
    covered = count >= 2
    value = np.where(covers, planar, 0.0).sum(axis=0) / np.maximum(count, 1) / cosine
    return np.where(covered, value, fallback), covered


def score(root):
    prereg = json.loads((root / "preregistration.json").read_text())
    if sha(root / "reused_seals.json") != prereg["root_files"]["reused_seals.json"]:
        raise PermissionError("reused_seals.json changed since the lock")
    reused = json.loads((root / "reused_seals.json").read_text())
    lock_path = root / "eval_OVERLAP/frame_role_lock.json"
    if sha(lock_path) != reused["frame_role_lock"]["sha256"]:
        raise PermissionError("frame-role lock changed")
    lock = json.loads(lock_path.read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    if sha(reused["arms_seal"]["path"]) != reused["arms_seal"]["sha256"]:
        raise PermissionError("the reused arms seal changed")
    arms = json.loads(Path(reused["arms_seal"]["path"]).read_text())
    seals = {}
    for name, parts in reused["jo_seals"].items():
        for part in parts:
            if sha(part["path"]) != part["sha256"]:
                raise PermissionError(f"a reused seal changed: {part['path']}")
        seals[name] = ds_seal([p["path"] for p in parts], keys)
    checkpoints = {s: sha(root / f"checkpoints/{s}/depthsplat_jepa_final.pth") for s in BUILDERS["JOL"]}
    for name in BUILDERS["JOL"]:
        seals[name] = ds_seal(sorted((root / f"ds/{COHORT}/{name}").glob("ds_seal_*of*.json")), keys, checkpoints[name])
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        if record["source_cohort"] != SOURCE:
            continue
        frames = {f["frame_id"]: f for f in record["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            p = load(arms["predictions"][f"{COHORT}/{key}"])
            cosine = ray_cosine(record, qid)
            near = p["near"].astype(bool)
            zero_hit = not near.any()
            h3 = p["REPROJ_HARMONIC"]
            g1n = np.mean([np.where(p[f"OPACITY_G1_{s}"] >= 0.5,
                                    p[f"G1_{s}"] / np.maximum(p[f"OPACITY_G1_{s}"].astype(np.float64), 1e-6), h3)
                           for s in arms["seeds"]], axis=0)
            built, cover = {}, {}
            for name, seeds in BUILDERS.items():
                built[name], cover[name] = pooled([load(seals[s][key]) for s in seeds], cosine, h3)
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            far = valid & ~near
            regions = {"FARH": far if not zero_hit else np.zeros_like(far), "FAR": far, "NEAR": valid & near}
            values = {"zero_hit": zero_hit}
            for n in BUILDERS:
                values[f"{n}_farh_coverage"] = float(cover[n][far].mean()) if far.any() and not zero_hit else None
            for name, x in {"H3": h3, "G1N": g1n, "JO": built["JO"], "JOL": built["JOL"]}.items():
                for region, mask in regions.items():
                    values[f"{name}|{region}"] = (float(np.mean(np.abs(x[mask] - gt[mask]) / gt[mask]))
                                                  if mask.any() else None)
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "values": values})
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and not isinstance(v, bool) and np.isfinite(v):
                grouped[name][row["scene_id"]][row["role"]].append(v)
    table = {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "zero_hit_queries": sum(r["values"]["zero_hit"] for r in rows),
                                            "query_depth_reads": access, "checkpoints": checkpoints,
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "VAL80 queries;", sum(r["values"]["zero_hit"] for r in rows), "without a context hit")


PRIMARY = {"P1_MORE_DATA_COMPLETION_FARH": ("JO|FARH", "JOL|FARH"),
           "P2_SCALED_COMPLETION_VS_HARMONIC_FARH": ("H3|FARH", "JOL|FARH")}
SECONDARY = {"S1_ORIGINAL_COMPLETION_VS_HARMONIC_FARH": ("H3|FARH", "JO|FARH"),
             "S2_SCALED_COMPLETION_VS_G1N_FARH": ("G1N|FARH", "JOL|FARH"),
             "S3_MORE_DATA_NEAR": ("JO|NEAR", "JOL|NEAR"),
             "S4_MORE_DATA_FAR_ALL_QUERIES": ("JO|FAR", "JOL|FAR")}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {
        "P1": label(primary["P1_MORE_DATA_COMPLETION_FARH"]["status"],
                    ("MORE_DATA_HELPS_COMPLETION", "MORE_DATA_HELPS_COMPLETION_PARTIAL",
                     "MORE_DATA_COMPLETION_NOT_ESTABLISHED", "MORE_DATA_HURTS_COMPLETION")),
        "P2": label(primary["P2_SCALED_COMPLETION_VS_HARMONIC_FARH"]["status"],
                    ("SCALED_COMPLETION_BEATS_HARMONIC", "SCALED_COMPLETION_PARTIAL",
                     "SCALED_COMPLETION_NOT_ESTABLISHED", "SCALED_COMPLETION_HARMFUL")),
    }
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    outputs = Path(__file__).resolve().parents[2]
    earlier = json.loads((outputs / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1/results/ARKIT160_OVERLAP_results.json")
                         .read_text())["by_cohort"][SOURCE]
    replication = {k: abs(means[k]["mean"] - earlier[k]) for k in ("H3|FARH", "JO|FARH")}
    write(root / "results/OS_results.json", {"cohort": f"{COHORT} ({SOURCE} part)", "branches": branches,
                                             "primary": primary, "secondary": secondary, "means": means,
                                             "replication_abs_diff": replication})
    print("ANALYZED", branches)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} ({s['scenes']} scenes) {s['status']}")
    for k, v in means.items():
        print(f"  {k:22s} {v['mean']:.4f} ({v['scenes']})")
    print("  replication:", json.dumps({k: round(v, 12) for k, v in replication.items()}))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
