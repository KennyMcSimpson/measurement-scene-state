"""EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1: scoring and the preregistered analysis on ARKIT_SELECT (80 fresh scans).

score: every prediction must be sealed and unchanged before query depth is read: the arms seal (REPROJ_HARMONIC, NEAR,
depth and opacity of G1 and GA for three seeds), the DepthSplat CTX seals of JO1-JO3 (cap 3.0) and the selector seal
(context depth spread S per episode). Per query with at least one context hit:
  H3        REPROJ_HARMONIC
  GAN, G1N  per seed depth / opacity where opacity >= 0.5, else H3; mean over the three seeds
  JO        where at least two JO seeds cover (alpha >= 0.5) the mean ray depth of the covering seeds, else H3
  HGAN      (H3 + GAN) / 2
  SEL_X     X where the episode's S >= 1.2, else H3 (X = GAN, G1N, JO)
  DSEL_GAN  GAN where the query's median |log(GAN / H3)| outside NEAR >= 0.18, else H3
AbsRel on FARH (valid pixels outside the 8-px NEAR). Scene value = mean over roles of the mean over queries.
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate (ties count as non-worse).
Run through cv_host_run.py from the repository root (CUDA hidden):  python sel_score.py ROOT score | analyze
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

COHORT, ALPHA_MIN, OPACITY_MIN, THRESHOLD, DISAGREE = "ARKIT_SELECT", 0.5, 0.5, 1.2, 0.18
JO_SEEDS = ("JO1", "JO2", "JO3")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def ds_seal(root, name, keys):
    preds = {}
    for part in sorted((root / f"ds/{COHORT}" / name).glob("ds_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != COHORT or seal["scale_max"] != 3.0:
            raise PermissionError(f"{part}: not a complete sealed run")
        preds.update(seal["predictions"])
    if set(preds) != set(keys):
        raise ValueError(f"{name}: sealed queries differ from the frame-role lock")
    return preds


def ray_cosine(record, qid):
    from mcss.measurement_planning import geometry as g
    from mcss.measurement_planning.information import TeacherAccess

    teacher = TeacherAccess(record, [qid], (128, 160), Path.cwd(), [])
    return np.asarray(g.ray_cosine(teacher.query_camera(qid)), dtype=np.float64)


def normalized(p, arm, seeds, fallback):
    return np.mean([np.where(p[f"OPACITY_{arm}_{s}"] >= OPACITY_MIN,
                             p[f"{arm}_{s}"] / np.maximum(p[f"OPACITY_{arm}_{s}"].astype(np.float64), 1e-6), fallback)
                    for s in seeds], axis=0)


def score(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    prereg = json.loads((root / "preregistration.json").read_text())
    if sha(root / "selector_seal.json") != prereg["root_files"]["selector_seal.json"]:
        raise PermissionError("selector seal changed since the lock")
    selector = json.loads((root / "selector_seal.json").read_text())["episodes"]
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    arms = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    if arms["query_depth_read"] is not False or any(f"{COHORT}/{k}" not in arms["predictions"] for k in keys):
        raise PermissionError("arms seal incomplete or not certified")
    jo = {s: ds_seal(root, s, keys) for s in JO_SEEDS}
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        record = records[e["scene_id"]]
        frames = {f["frame_id"]: f for f in record["frames"]}
        use = selector[f"{e['scene_id']}/{e['role']}"]["use_learned"]
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            p = load(arms["predictions"][f"{COHORT}/{key}"])
            near = p["near"].astype(bool)
            if not near.any():
                rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "zero_hit": True, "values": {}})
                continue
            h3 = p["REPROJ_HARMONIC"]
            gan, g1n = normalized(p, "GA", arms["seeds"], h3), normalized(p, "G1", arms["seeds"], h3)
            cosine = ray_cosine(record, qid)
            arrays = [load(jo[s][key]) for s in JO_SEEDS]
            covers = np.array([a["CTX_alpha"].astype(np.float64) >= ALPHA_MIN for a in arrays])
            ray = np.array([a["CTX_depth_z"].astype(np.float64) / np.maximum(a["CTX_alpha"].astype(np.float64), 1e-12)
                            for a in arrays]) / cosine
            count = covers.sum(axis=0)
            jo_pred = np.where(count >= 2, np.where(covers, ray, 0.0).sum(axis=0) / np.maximum(count, 1), h3)
            far = ~near
            disagree = (float(np.median(np.abs(np.log(np.maximum(gan[far], 1e-6) / h3[far])))) if far.any()
                        else None)
            readouts = {"H3": h3, "GAN": gan, "G1N": g1n, "JO": jo_pred, "HGAN": (h3 + gan) / 2,
                        "SEL_GAN": gan if use else h3, "SEL_G1N": g1n if use else h3, "SEL_JO": jo_pred if use else h3,
                        "DSEL_GAN": gan if disagree is not None and disagree >= DISAGREE else h3}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            farh = np.isfinite(gt) & (gt > 0) & far
            values = {f"{n}|FARH": (float(np.mean(np.abs(x[farh] - gt[farh]) / gt[farh])) if farh.any() else None)
                      for n, x in readouts.items()}
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "zero_hit": False,
                         "use_learned": use, "spread": selector[f"{e['scene_id']}/{e['role']}"]["spread"],
                         "disagree": disagree, "values": values})
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and np.isfinite(v):
                grouped[name][row["scene_id"]][row["role"]].append(v)
    table = {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "zero_hit_queries": sum(r["zero_hit"] for r in rows),
                                            "query_depth_reads": access, "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries;", sum(r["zero_hit"] for r in rows), "without a context hit")


PRIMARY = {"P1_SELECTED_CARRIER_VS_HARMONIC_FARH": ("H3|FARH", "SEL_GAN|FARH")}
SECONDARY = {"S1_SELECTED_HYPERSIM_CARRIER_FARH": ("H3|FARH", "SEL_G1N|FARH"),
             "S2_SELECTED_DEPTHSPLAT_JEPA_FARH": ("H3|FARH", "SEL_JO|FARH"),
             "S3_CARRIER_UNSELECTED_FARH": ("H3|FARH", "GAN|FARH"), "S4_HYPERSIM_CARRIER_UNSELECTED_FARH": ("H3|FARH", "G1N|FARH"),
             "S5_DEPTHSPLAT_JEPA_UNSELECTED_FARH": ("H3|FARH", "JO|FARH"),
             "S6_COST_OF_SELECTION_FARH": ("GAN|FARH", "SEL_GAN|FARH"),
             "S7_CARRIER_ENSEMBLE_FARH": ("H3|FARH", "HGAN|FARH"),
             "S8_DISAGREEMENT_SELECTOR_FARH": ("H3|FARH", "DSEL_GAN|FARH")}


def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    status = primary["P1_SELECTED_CARRIER_VS_HARMONIC_FARH"]["status"]
    branch = {"SUPPORTED": "SELECTOR_CONSISTENT", "PARTIAL": "SELECTOR_PARTIAL",
              "HARMFUL": "SELECTOR_HARMFUL"}.get(status, "SELECTOR_NOT_ESTABLISHED")
    rows = [r for r in json.loads((root / "scores/query_rows.json").read_text()) if not r["zero_hit"]]
    episodes = {(r["scene_id"], r["role"]): r["use_learned"] for r in rows}
    gain = defaultdict(lambda: defaultdict(list))
    spread = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["values"]["GAN|FARH"] is not None:
            gain[r["scene_id"]][r["role"]].append(r["values"]["H3|FARH"] - r["values"]["GAN|FARH"])
            spread[r["scene_id"]][r["role"]].append(r["spread"])
    s_ids = sorted(gain)
    g = np.array([np.mean([np.mean(v) for v in gain[s].values()]) for s in s_ids])
    x = np.array([np.mean([np.mean(v) for v in spread[s].values()]) for s in s_ids])
    descriptive = {"episodes_with_hit_queries": len(episodes), "episodes_selected": int(sum(episodes.values())),
                   "spearman_scene_spread_vs_gain_GAN": float(np.corrcoef(rank(x), rank(g))[0, 1])}
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    write(root / "results/ARKIT_SELECT_results.json", {"cohort": COHORT, "branch": branch, "primary": primary,
                                                       "secondary": secondary, "means": means,
                                                       "descriptive": descriptive})
    print("ANALYZED", branch)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} ({s['scenes']} scenes) {s['status']}")
    for k, v in means.items():
        print(f"  {k:16s} {v['mean']:.4f} ({v['scenes']})")
    print("  descriptive:", json.dumps(descriptive))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"score": score, "analyze": analyze}[stage](root)
