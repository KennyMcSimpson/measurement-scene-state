"""EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1: threshold selection on DEV, scoring and the preregistered analysis.

Every reused seal is hash-checked (reused_seals.json) before any query depth is read. Readouts per query:
  ZS_CACHE(t)  ray depth (depth_z / alpha / cos) of the released CACHE state where alpha >= t, else REPROJ7_HARMONIC (R7)
  ZS_TTT(t)    the same for the TTT_m1 state;   CORE_B(t) = sqrt(R7 x ZS_TTT(t))
  JNS / JLS    mean over JN1-JN3 / JL1-JL3 of the per-query AbsRel of their CACHE state at t = 0.5
AbsRel on ALL, NEAR7, FAR7; coverage = share of valid pixels with alpha >= t. Scene value = mean over roles of the mean
over queries.
select:  DEV cohorts only (ARKIT_SYSTEM, ARKIT_SYSTEM2), grid 0.5 0.6 0.7 0.8 0.9 0.95 0.99; tau_C = argmin of the DEV
         scene mean of ZS_CACHE(t)|ALL, tau_T = argmin of CORE_B(t)|ALL (ties: the smaller t). Writes selection.json and
         audit/selection_locked_at.txt; refuses if either exists or if confirmation scores exist.
score:   CONFIRM cohorts only (ARKIT_SYSTEM3, ARKIT_VAL), after the selection: readouts at 0.5 and at the selected tau.
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python at_score.py ROOT select | score | analyze
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

GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
REGIONS = ("ALL", "NEAR7", "FAR7")
SEED_GROUPS = {"JNS": ("JN1", "JN2", "JN3"), "JLS": ("JL1", "JL2", "JL3")}


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
    """Yields (scene, role, key, r7, near7, gt, valid, arrays) after every seal of the cohort is verified."""
    entry = json.loads((root / "reused_seals.json").read_text())[cohort]
    for item in [entry["frame_role_lock"], entry["geometry"], *[p for parts in entry["seals"].values() for p in parts]]:
        if sha(item["path"]) != item["sha256"]:
            raise PermissionError(f"a reused file changed: {item['path']}")
    lock = json.loads(Path(entry["frame_role_lock"]["path"]).read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    geometry = json.loads(Path(entry["geometry"]["path"]).read_text())
    seals = {}
    for name, parts in entry["seals"].items():
        seals[name] = {}
        for part in parts:
            seal = json.loads(Path(part["path"]).read_text())
            if seal["query_depth_read"] is not False:
                raise PermissionError(f"{part['path']}: not sealed")
            seals[name].update(seal["predictions"])
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo = load(geometry["predictions"][key])
            arrays = {name: load(preds[key]) for name, preds in seals.items()}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"cohort": cohort, "key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            yield (e["scene_id"], e["role"], key, geo["REPROJ7_HARMONIC"], geo["near7"].astype(bool), geo["cosine"], gt,
                   valid, arrays)


def readout(arrays, arm, threshold, r7, cosine):
    alpha = arrays[f"{arm}_alpha"].astype(np.float64)
    ray = arrays[f"{arm}_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
    return np.where(alpha >= threshold, ray, r7), alpha >= threshold


def gm(a, b):
    return np.sqrt(np.maximum(a, 1e-6) * np.maximum(b, 1e-6))


def regional(values, name, p, gt, valid, near7):
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
    if (root / "selection.json").exists() or (root / "audit/selection_locked_at.txt").exists():
        raise SystemExit("a selection exists already")
    if (root / "scores").exists():
        raise SystemExit("confirmation scores exist")
    rows, access, started = [], [], time.time()
    for cohort in ("ARKIT_SYSTEM", "ARKIT_SYSTEM2"):
        for scene, role, key, r7, near7, cosine, gt, valid, arrays in queries(root, cohort, access):
            values = {}
            for t in GRID:
                cache, covered = readout(arrays["ZS"], "CACHE", t, r7, cosine)
                ttt, ttt_covered = readout(arrays["ZS_TTT"], "TTT_m1", t, r7, cosine)
                for name, p in ((f"ZS_CACHE@{t}", cache), (f"ZS_TTT@{t}", ttt), (f"CORE_B@{t}", gm(r7, ttt))):
                    values[f"{name}|ALL"] = float(np.mean(np.abs(p[valid] - gt[valid]) / gt[valid]))
                values[f"ZS_CACHE@{t}_coverage"] = float(covered[valid].mean())
                values[f"ZS_TTT@{t}_coverage"] = float(ttt_covered[valid].mean())
            rows.append({"cohort": "DEV", "scene_id": scene, "role": role, "query": key, "values": values})
    table = scene_table(rows)
    curve = {str(t): {k: float(np.mean(list(table[f"DEV:{name}@{t}{suffix}"].values())))
                      for k, name, suffix in (("cache", "ZS_CACHE", "|ALL"), ("ttt", "ZS_TTT", "|ALL"),
                                              ("core_b", "CORE_B", "|ALL"), ("cache_coverage", "ZS_CACHE", "_coverage"),
                                              ("ttt_coverage", "ZS_TTT", "_coverage"))}
             for t in GRID}
    pick = lambda k: min(GRID, key=lambda t: (curve[str(t)][k], t))  # noqa: E731
    selection = {"grid": list(GRID), "dev_cohorts": ["ARKIT_SYSTEM", "ARKIT_SYSTEM2"],
                 "dev_scenes": len(table["DEV:ZS_CACHE@0.5|ALL"]), "dev_queries": len(rows), "dev_curve": curve,
                 "tau_cache": pick("cache"), "tau_core_b": pick("core_b"), "confirmation_query_depth_read": False,
                 "dev_query_depth_reads": len(access), "seconds": round(time.time() - started, 1)}
    write(root / "selection.json", selection)
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/selection_locked_at.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} "
                                                        f"sha256(selection.json)={sha(root / 'selection.json')}\n")
    print("SELECTED tau_cache", selection["tau_cache"], "tau_core_b", selection["tau_core_b"])
    for t in GRID:
        c = curve[str(t)]
        print(f"  t={t}: ZS_CACHE {c['cache']:.4f} (cov {c['cache_coverage']:.2f})  ZS_TTT {c['ttt']:.4f} "
              f"(cov {c['ttt_coverage']:.2f})  CORE_B {c['core_b']:.4f}")


def score(root):
    stamp = (root / "audit/selection_locked_at.txt").read_text()
    if f"sha256(selection.json)={sha(root / 'selection.json')}" not in stamp:
        raise PermissionError("selection.json changed after the selection")
    selection = json.loads((root / "selection.json").read_text())
    tau_c, tau_t = selection["tau_cache"], selection["tau_core_b"]
    rows, access, started = [], [], time.time()
    for cohort in ("ARKIT_SYSTEM3", "ARKIT_VAL"):
        for scene, role, key, r7, near7, cosine, gt, valid, arrays in queries(root, cohort, access):
            values = {}
            for name, threshold, source, arm in (("ZS_CACHE_05", 0.5, "ZS", "CACHE"), ("ZS_CACHE_TAU", tau_c, "ZS", "CACHE"),
                                                 ("ZS_TTT_05", 0.5, "ZS_TTT", "TTT_m1"),
                                                 ("ZS_TTT_TAU", tau_t, "ZS_TTT", "TTT_m1")):
                p, covered = readout(arrays[source], arm, threshold, r7, cosine)
                regional(values, name, p, gt, valid, near7)
                values[f"{name}_coverage"] = float(covered[valid].mean()) if valid.any() else None
                if name.startswith("ZS_TTT"):
                    regional(values, name.replace("ZS_TTT", "CORE_B"), gm(r7, p), gt, valid, near7)
            regional(values, "R7", r7, gt, valid, near7)
            for group, seeds in SEED_GROUPS.items():
                per_seed = []
                for s in seeds:
                    seed_values = {}
                    regional(seed_values, s, readout(arrays[s], "CACHE", 0.5, r7, cosine)[0], gt, valid, near7)
                    per_seed.append(seed_values)
                for region in REGIONS:
                    parts = [v[f"{s}|{region}"] for v, s in zip(per_seed, seeds)]
                    values[f"{group}_CACHE|{region}"] = None if any(v is None for v in parts) else float(np.mean(parts))
            rows.append({"cohort": cohort, "scene_id": scene, "role": role, "query": key, "values": values})
    table = scene_table(rows)
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access, "tau_cache": tau_c,
                                            "tau_core_b": tau_t, "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries at tau_cache", tau_c, "tau_core_b", tau_t)


S3, VAL = "ARKIT_SYSTEM3", "ARKIT_VAL"
PRIMARY = {"P1_JEPA_VS_THRESHOLD_SYSTEM3_ALL": (f"{S3}:ZS_CACHE_TAU|ALL", f"{S3}:JNS_CACHE|ALL"),
           "P2_THRESHOLD_CORE_B_SYSTEM3_ALL": (f"{S3}:CORE_B_05|ALL", f"{S3}:CORE_B_TAU|ALL")}
SECONDARY = {"S1_THRESHOLD_FEEDFORWARD_SYSTEM3_ALL": (f"{S3}:ZS_CACHE_05|ALL", f"{S3}:ZS_CACHE_TAU|ALL"),
             "S2_SCALED_JEPA_VS_THRESHOLD_SYSTEM3_ALL": (f"{S3}:ZS_CACHE_TAU|ALL", f"{S3}:JLS_CACHE|ALL"),
             "S3_JEPA_VS_THRESHOLD_VAL_ALL": (f"{VAL}:ZS_CACHE_TAU|ALL", f"{VAL}:JNS_CACHE|ALL"),
             "S4_THRESHOLD_CORE_B_VAL_ALL": (f"{VAL}:CORE_B_05|ALL", f"{VAL}:CORE_B_TAU|ALL"),
             "S5_THRESHOLD_FEEDFORWARD_VAL_ALL": (f"{VAL}:ZS_CACHE_05|ALL", f"{VAL}:ZS_CACHE_TAU|ALL"),
             "S6_JEPA_VS_THRESHOLD_SYSTEM3_NEAR7": (f"{S3}:ZS_CACHE_TAU|NEAR7", f"{S3}:JNS_CACHE|NEAR7"),
             "S7_THRESHOLD_TTT_SYSTEM3_ALL": (f"{S3}:ZS_TTT_05|ALL", f"{S3}:ZS_TTT_TAU|ALL")}


def label(status, names):
    return {"SUPPORTED": names[0], "PARTIAL": names[1], "HARMFUL": names[3]}.get(status, names[2])


def safe_contrast(table, left, right):
    """contrast(), except that two identical readouts (a selected tau of 0.5) give a fixed NOT_ESTABLISHED record."""
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
    branches = {
        "P1": label(primary["P1_JEPA_VS_THRESHOLD_SYSTEM3_ALL"]["status"],
                    ("JEPA_BEYOND_THRESHOLD", "JEPA_BEYOND_THRESHOLD_PARTIAL", "JEPA_VS_THRESHOLD_NOT_ESTABLISHED",
                     "THRESHOLD_BEATS_JEPA")),
        "P2": ("THRESHOLD_CORE_B_NOT_ESTABLISHED" if selection["tau_core_b"] == 0.5 else
               label(primary["P2_THRESHOLD_CORE_B_SYSTEM3_ALL"]["status"],
                     ("THRESHOLD_IMPROVES_CORE_B", "THRESHOLD_IMPROVES_CORE_B_PARTIAL", "THRESHOLD_CORE_B_NOT_ESTABLISHED",
                      "THRESHOLD_HARMS_CORE_B"))),
    }
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    write(root / "results/AT_results.json", {"selection": {k: selection[k] for k in ("tau_cache", "tau_core_b")},
                                             "branches": branches, "primary": primary, "secondary": secondary,
                                             "means": means})
    print("ANALYZED", branches, "tau_cache", selection["tau_cache"], "tau_core_b", selection["tau_core_b"])
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} {s['status']}")
    for k, v in means.items():
        if k.endswith("|ALL") or k.endswith("coverage"):
            print(f"  {k:36s} {v['mean']:.4f} ({v['scenes']})")


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"select": select, "score": score, "analyze": analyze}[stage](root)
