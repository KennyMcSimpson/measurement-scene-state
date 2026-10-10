"""EXP-3D-RGBD-COREB-WEIGHT-ARKIT-V1: weight selection on DEV, scoring and the preregistered analysis.

Every reused file is hash-checked (reused_seals.json) before any query depth is read. Readouts per query: R7 =
REPROJ7_HARMONIC; ZS_TTT_F7 = TTT_m1 ray depth (depth_z / alpha / cos) where alpha >= 0.5, else R7; GEO_w =
R7^(1-w) x ZS_TTT_F7^w (both floored at 1e-6); CORE_B = GEO_0.5. AbsRel on ALL, NEAR7, FAR7. Scene value = mean over
roles of the mean over queries.
select:  DEV (ARKIT_SYSTEM, ARKIT_SYSTEM2), grid 0.5 0.6 0.7 0.8 0.9 1.0: w* = argmin of the DEV scene mean of ALL
         (ties: the smaller w). Writes selection.json + audit/selection_locked_at.txt.
score:   CONFIRM (ARKIT_SYSTEM3, ARKIT_VAL) after the selection.
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python cw_score.py ROOT select | score | analyze
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

GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
REGIONS = ("ALL", "NEAR7", "FAR7")


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


def queries(root, cohort, access):
    entry = json.loads((root / "reused_seals.json").read_text())[cohort]
    for item in (entry["frame_role_lock"], entry["geometry"], *entry["zs"]):
        if sha(item["path"]) != item["sha256"]:
            raise PermissionError(f"a reused file changed: {item['path']}")
    lock = json.loads(Path(entry["frame_role_lock"]["path"]).read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    geometry = json.loads(Path(entry["geometry"]["path"]).read_text())
    zs = {}
    for part in entry["zs"]:
        zs.update(json.loads(Path(part["path"]).read_text())["predictions"])
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo, z = load(geometry["predictions"][key]), load(zs[key])
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"cohort": cohort, "key": key, "path": frames[qid]["depth"]})
            r7, cosine = geo["REPROJ7_HARMONIC"], geo["cosine"]
            alpha = z["TTT_m1_alpha"].astype(np.float64)
            ray = z["TTT_m1_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
            yield e["scene_id"], e["role"], key, r7, np.where(alpha >= 0.5, ray, r7), geo["near7"].astype(bool), gt


def geo_w(r7, ttt, w):
    return np.exp((1 - w) * np.log(np.maximum(r7, 1e-6)) + w * np.log(np.maximum(ttt, 1e-6)))


def regional(values, name, p, gt, near7):
    valid = np.isfinite(gt) & (gt > 0)
    for region, mask in zip(REGIONS, (valid, valid & near7, valid & ~near7)):
        values[f"{name}|{region}"] = float(np.mean(np.abs(p[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def scene_table(rows):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and np.isfinite(v):
                grouped[f"{row['cohort']}:{name}"][row["scene_id"]][row["role"]].append(v)
    return {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}


def select(root):
    if (root / "selection.json").exists() or (root / "audit/selection_locked_at.txt").exists() or (root / "scores").exists():
        raise SystemExit("a selection or a confirmation score exists already")
    rows, access, started = [], [], time.time()
    for cohort in ("ARKIT_SYSTEM", "ARKIT_SYSTEM2"):
        for scene, role, key, r7, ttt, near7, gt in queries(root, cohort, access):
            values = {}
            for w in GRID:
                regional(values, f"GEO_{w}", geo_w(r7, ttt, w), gt, near7)
            regional(values, "R7", r7, gt, near7)
            rows.append({"cohort": "DEV", "scene_id": scene, "role": role, "query": key, "values": values})
    table = scene_table(rows)
    curve = {str(w): float(np.mean(list(table[f"DEV:GEO_{w}|ALL"].values()))) for w in GRID}
    consistency = {str(w): int(sum(table["DEV:R7|ALL"][s] > table[f"DEV:GEO_{w}|ALL"][s] for s in table["DEV:R7|ALL"]))
                   for w in GRID}
    best = min(GRID, key=lambda w: (curve[str(w)], w))
    selection = {"grid": list(GRID), "dev_cohorts": ["ARKIT_SYSTEM", "ARKIT_SYSTEM2"], "dev_scenes": len(table["DEV:R7|ALL"]),
                 "dev_curve_all": curve, "dev_scenes_better_than_r7": consistency,
                 "dev_r7_all": float(np.mean(list(table["DEV:R7|ALL"].values()))), "w_star": best,
                 "confirmation_query_depth_read": False, "seconds": round(time.time() - started, 1)}
    write(root / "selection.json", selection)
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/selection_locked_at.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} "
                                                        f"sha256(selection.json)={sha(root / 'selection.json')}\n")
    print("SELECTED w*", best, "R7", round(selection["dev_r7_all"], 4))
    for w in GRID:
        print(f"  w={w}: {curve[str(w)]:.4f}  scenes better than R7 {consistency[str(w)]}/{selection['dev_scenes']}")


def score(root):
    stamp = (root / "audit/selection_locked_at.txt").read_text()
    if f"sha256(selection.json)={sha(root / 'selection.json')}" not in stamp:
        raise PermissionError("selection.json changed after the selection")
    w_star = json.loads((root / "selection.json").read_text())["w_star"]
    rows, access, started = [], [], time.time()
    for cohort in ("ARKIT_SYSTEM3", "ARKIT_VAL"):
        for scene, role, key, r7, ttt, near7, gt in queries(root, cohort, access):
            values = {}
            for name, p in (("R7", r7), ("ZS_TTT_F7", ttt), ("CORE_B", geo_w(r7, ttt, 0.5)), ("GEO_STAR", geo_w(r7, ttt, w_star))):
                regional(values, name, p, gt, near7)
            rows.append({"cohort": cohort, "scene_id": scene, "role": role, "query": key, "values": values})
    table = scene_table(rows)
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access, "w_star": w_star,
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries at w*", w_star)


S3, VAL = "ARKIT_SYSTEM3", "ARKIT_VAL"
PRIMARY = {"P1_WEIGHT_VS_CORE_B_SYSTEM3_ALL": (f"{S3}:CORE_B|ALL", f"{S3}:GEO_STAR|ALL"),
           "P2_WEIGHT_VS_CORE_B_VAL_ALL": (f"{VAL}:CORE_B|ALL", f"{VAL}:GEO_STAR|ALL")}
SECONDARY = {"S1_WEIGHT_VS_MODEL_FREE_SYSTEM3_ALL": (f"{S3}:R7|ALL", f"{S3}:GEO_STAR|ALL"),
             "S2_WEIGHT_VS_MODEL_FREE_VAL_ALL": (f"{VAL}:R7|ALL", f"{VAL}:GEO_STAR|ALL"),
             "S3_TTT_ALONE_VS_MODEL_FREE_SYSTEM3_ALL": (f"{S3}:R7|ALL", f"{S3}:ZS_TTT_F7|ALL"),
             "S4_TTT_ALONE_VS_CORE_B_SYSTEM3_ALL": (f"{S3}:CORE_B|ALL", f"{S3}:ZS_TTT_F7|ALL"),
             "S5_WEIGHT_VS_CORE_B_SYSTEM3_NEAR7": (f"{S3}:CORE_B|NEAR7", f"{S3}:GEO_STAR|NEAR7"),
             "S6_WEIGHT_VS_CORE_B_SYSTEM3_FAR7": (f"{S3}:CORE_B|FAR7", f"{S3}:GEO_STAR|FAR7")}
NAMES = ("WEIGHT_IMPROVES_CORE_B", "WEIGHT_IMPROVES_CORE_B_PARTIAL", "WEIGHT_CORE_B_NOT_ESTABLISHED", "WEIGHT_HARMS_CORE_B")


def safe_contrast(table, left, right):
    """contrast(), except that two identical readouts (w* = 0.5) give a fixed NOT_ESTABLISHED record."""
    common = sorted(set(table[left]) & set(table[right]))
    if all(table[left][s] == table[right][s] for s in common):
        return {"left": left, "right": right, "scenes": len(common), "status": "NOT_ESTABLISHED", "identical": True,
                "mean": 0.0, "ci95": [0.0, 0.0], "positive": 0, "tied": len(common), "negative": 0, "median": 0.0}
    return contrast(table, left, right)


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    primary = {k: safe_contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: safe_contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    label = lambda s: {"SUPPORTED": NAMES[0], "PARTIAL": NAMES[1], "HARMFUL": NAMES[3]}.get(s, NAMES[2])  # noqa: E731
    branches = {"P1": label(primary["P1_WEIGHT_VS_CORE_B_SYSTEM3_ALL"]["status"]),
                "P2": label(primary["P2_WEIGHT_VS_CORE_B_VAL_ALL"]["status"])}
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    write(root / "results/CW_results.json", {"w_star": selection["w_star"], "branches": branches, "primary": primary,
                                             "secondary": secondary, "means": means})
    print("ANALYZED", branches, "w*", selection["w_star"])
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} {s['status']}")
    for k, v in means.items():
        print(f"  {k:28s} {v['mean']:.4f} ({v['scenes']})")


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"select": select, "score": score, "analyze": analyze}[stage](root)
