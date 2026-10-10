"""EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1: scoring and the preregistered analysis on EVAL34.

score: every prediction must be sealed and unchanged (arms seal: REPROJ_HARMONIC, NEAR, G1 for three seeds; one
DepthSplat seal per reverse-JEPA seed J1-J3, CTX arm); only then is query depth read. Per query, on FAR (valid pixels
outside the 8-px NEAR of the context reprojection):
  H3   REPROJ_HARMONIC
  G1   mean of the three G1 seeds' depth
  J    where at least two of the three J seeds cover the pixel (alpha >= 0.5): the mean ray depth (depth_z / alpha /
       cos) over the covering seeds; elsewhere H3
  JG   covered: (J + G1) / 2; elsewhere G1
  HG   (H3 + G1) / 2
  HJ   covered: (H3 + J) / 2; elsewhere H3
  HJG  covered: (H3 + J + G1) / 3; elsewhere (H3 + G1) / 2
  HG_GEO, HJG_GEO  the same with geometric means
AbsRel on FAR and on ALL with NEAR taken from H3 (X_ALL = NEAR ? H3 : X). Scene value = mean over roles of the mean
over queries (queries without FAR pixels skipped).
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):
  python ens_score.py ROOT score | analyze
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

COHORT, SEEDS, ALPHA_MIN = "EVAL34", ("J1", "J2", "J3"), 0.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def absrel(p, gt, mask):
    return float(np.mean(np.abs(p[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def score(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    if arms["query_depth_read"] is not False or any(f"{COHORT}/{k}" not in arms["predictions"] for k in keys):
        raise PermissionError("arms seal incomplete or not certified")
    js = {}
    for s in SEEDS:
        preds = {}
        for part in sorted((root / f"ds/{COHORT}" / s).glob("ds_seal_*of*.json")):
            seal = json.loads(part.read_text())
            if seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT:
                raise PermissionError(f"{part}: not a complete sealed run")
            preds.update(seal["predictions"])
        if set(preds) != set(keys):
            raise ValueError(f"{s}: sealed queries differ from the frame-role lock")
        js[s] = preds
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            sealed = load(arms["predictions"][f"{COHORT}/{key}"])
            arrays = [load(js[s][key]) for s in SEEDS]
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            near = sealed["near"].astype(bool)
            far = valid & ~near
            h3 = sealed["REPROJ_HARMONIC"]
            g1 = np.mean([sealed[f"G1_{s}"] for s in arms["seeds"]], axis=0)
            covers, depths = [], []
            for a in arrays:
                alpha = a["CTX_alpha"].astype(np.float64)
                covers.append(alpha >= ALPHA_MIN)
                depths.append(a["CTX_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12))
            cosine = ray_cosine(records[e["scene_id"]], qid)
            covers = np.array(covers)
            count = covers.sum(axis=0)
            covered = count >= 2
            planar = np.where(covers, np.array(depths), 0.0).sum(axis=0) / np.maximum(count, 1)
            j = np.where(covered, planar / cosine, h3)
            geo = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 1e-6)) for x in xs], axis=0))  # noqa: E731
            readouts = {"H3": h3, "G1": g1, "J": j, "JG": np.where(covered, (j + g1) / 2, g1), "HG": (h3 + g1) / 2,
                        "HJ": np.where(covered, (h3 + j) / 2, h3),
                        "HJG": np.where(covered, (h3 + j + g1) / 3, (h3 + g1) / 2),
                        "HG_GEO": geo(h3, g1), "HJG_GEO": np.where(covered, geo(h3, j, g1), geo(h3, g1))}
            values = {"J_far_coverage": float(covered[far].mean()) if far.any() else None}
            for name, p in readouts.items():
                values[f"{name}|FAR"] = absrel(p, gt, far)
                values[f"{name}|ALL"] = absrel(np.where(near, h3, p), gt, valid)
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "values": values})
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and np.isfinite(v):
                grouped[name][row["scene_id"]][row["role"]].append(v)
    table = {n: {s: float(np.mean([np.mean(v) for v in roles.values()])) for s, roles in sc.items()}
             for n, sc in grouped.items()}
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access,
                                            "arms_seal_sha256": sha(root / f"raw/eval/{COHORT}/prediction_seal.json"),
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries")


def ray_cosine(record, qid):
    """The frozen ray cosine of the query camera (planar -> ray distance), as V1's geometry seal computes it."""
    from mcss.measurement_planning import geometry as g
    from mcss.measurement_planning.information import TeacherAccess

    teacher = TeacherAccess(record, [qid], (128, 160), Path.cwd(), [])
    return np.asarray(g.ray_cosine(teacher.query_camera(qid)), dtype=np.float64)


PRIMARY = {"E1_ENSEMBLE_VS_HARMONIC_FAR": ("H3|FAR", "HJG|FAR"),
           "E2_JEPA_ADDS_TO_ENSEMBLE_FAR": ("HG|FAR", "HJG|FAR")}
SECONDARY = {"S1_HG_VS_HARMONIC_FAR": ("H3|FAR", "HG|FAR"), "S2_JG_VS_HARMONIC_FAR": ("H3|FAR", "JG|FAR"),
             "S3_G1_VS_HARMONIC_FAR": ("H3|FAR", "G1|FAR"), "S4_J_VS_HARMONIC_FAR": ("H3|FAR", "J|FAR"),
             "S5_HJ_VS_HARMONIC_FAR": ("H3|FAR", "HJ|FAR"), "S6_JG_VS_G1_FAR": ("G1|FAR", "JG|FAR"),
             "S7_SHRINK_G1_FAR": ("G1|FAR", "HG|FAR"), "S8_HJG_VS_JG_FAR": ("JG|FAR", "HJG|FAR"),
             "S9_HJG_GEO_VS_HARMONIC_FAR": ("H3|FAR", "HJG_GEO|FAR"), "S10_HG_GEO_VS_HARMONIC_FAR": ("H3|FAR", "HG_GEO|FAR"),
             "S11_ENSEMBLE_VS_HARMONIC_ALL": ("H3|ALL", "HJG|ALL"), "S12_HG_VS_HARMONIC_ALL": ("H3|ALL", "HG|ALL")}


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    run = lambda spec: {k: contrast(table, a, b) for k, (a, b) in spec.items()}  # noqa: E731
    primary, secondary = run(PRIMARY), run(SECONDARY)
    e1, e2 = primary["E1_ENSEMBLE_VS_HARMONIC_FAR"]["status"], primary["E2_JEPA_ADDS_TO_ENSEMBLE_FAR"]["status"]
    branch_e1 = {"SUPPORTED": "ENSEMBLE_CONSISTENT", "PARTIAL": "ENSEMBLE_PARTIAL", "HARMFUL": "ENSEMBLE_HARMFUL"}.get(
        e1, "ENSEMBLE_NOT_ESTABLISHED")
    branch_e2 = {"SUPPORTED": "JEPA_ADDS", "PARTIAL": "JEPA_ADDS_PARTIAL", "HARMFUL": "JEPA_HARMFUL"}.get(e2, "JEPA_NO_ADD")
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    result = {"cohort": COHORT, "branch_e1": branch_e1, "branch_e2": branch_e2, "primary": primary,
              "secondary": secondary, "means": means}
    write(root / "results/EVAL34_ensemble_results.json", result)
    print("ANALYZED", branch_e1, branch_e2)
    for label, s in {**primary, **secondary}.items():
        print(f"  {label}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} {s['status']}")
    for k, v in means.items():
        print(f"  {k:22s} {v['mean']:.4f} ({v['scenes']})")


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
