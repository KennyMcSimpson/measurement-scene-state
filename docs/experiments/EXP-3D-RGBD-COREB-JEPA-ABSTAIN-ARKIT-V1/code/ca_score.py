"""EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1: candidate selection on DEV, scoring and the preregistered analysis.

Every reused seal is hash-checked (reused_seals.json) and every new JN1 seal must name JN1's checkpoint hash before any
query depth is read. Readouts per query (F7: depth_z / alpha / cos where alpha >= 0.5, else R7 = REPROJ7_HARMONIC):
  ZS_TTT_F7 (released TTT_m1 state), JN_F7 (a JN CACHE state), CORE_B = sqrt(R7 x ZS_TTT_F7) and the candidates
  MASK    sqrt(R7 x T'), T' = TTT ray depth where alpha_TTT >= 0.5 and alpha_JN >= 0.5, else R7
  TRIPLE  (R7 x ZS_TTT_F7 x JN_F7)^(1/3)
  PAIR    sqrt(ZS_TTT_F7 x JN_F7)
AbsRel on ALL, NEAR7, FAR7. Scene value = mean over roles of the mean over queries.
select:  DEV (ARKIT_SYSTEM, ARKIT_SYSTEM2, JN1): the candidate with the lowest DEV scene mean of ALL (ties: MASK, TRIPLE,
         PAIR); stop (NO_CANDIDATE_ON_DEV) unless it is below CORE_B. Writes selection.json + audit/selection_locked_at.txt.
score:   CONFIRM (ARKIT_SYSTEM3 with JN1-JN3, ARKIT_VAL with JN1), only after a selection that did not stop.
analyze: V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
Run through cv_host_run.py from the repository root (CUDA hidden):  python ca_score.py ROOT select | score | analyze
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

CANDIDATES = ("MASK", "TRIPLE", "PAIR")
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


def seal_files(root, cohort, entry, checkpoint):
    """Name -> {query key: item}: the reused seals (hash-checked) plus, for DEV, this experiment's JN1 CACHE seals."""
    files = {name: [p["path"] for p in parts] for name, parts in entry["seals"].items()}
    for parts in entry["seals"].values():
        for part in parts:
            if sha(part["path"]) != part["sha256"]:
                raise PermissionError(f"a reused seal changed: {part['path']}")
    if entry["role"] == "DEV":
        files["JN1"] = [str(p) for p in sorted((root / f"ds/{cohort}/JN1").glob("ds_seal_*of*.json"))]
    out = {}
    for name, paths in files.items():
        out[name] = {}
        for path in paths:
            seal = json.loads(Path(path).read_text())
            if seal["query_depth_read"] is not False or seal["limit"] or seal["cohort"] != cohort or seal["scale_max"] != 0.1:
                raise PermissionError(f"{path}: not a complete sealed run at cap 0.1")
            if entry["role"] == "DEV" and name == "JN1" and (seal["checkpoint_sha256"] != checkpoint
                                                              or seal["arms"] != ["CACHE"]):
                raise PermissionError(f"{path}: not JN1's CACHE run")
            out[name].update(seal["predictions"])
    return out


def queries(root, cohort, access):
    reused = json.loads((root / "reused_seals.json").read_text())
    entry = reused["cohorts"][cohort]
    for item in (entry["frame_role_lock"], entry["geometry"]):
        if sha(item["path"]) != item["sha256"]:
            raise PermissionError(f"a reused file changed: {item['path']}")
    lock = json.loads(Path(entry["frame_role_lock"]["path"]).read_text())["cohorts"][cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError(f"{cohort}: manifest changed since the frame-role lock")
    keys = {f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]}
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    geometry = json.loads(Path(entry["geometry"]["path"]).read_text())
    seals = seal_files(root, cohort, entry, reused["checkpoints"]["JN1"]["sha256"])
    for name, preds in seals.items():
        if set(preds) != keys:
            raise ValueError(f"{cohort}/{name}: sealed queries differ from the frame-role lock")
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            geo = load(geometry["predictions"][key])
            arrays = {name: load(preds[key]) for name, preds in seals.items()}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"cohort": cohort, "key": key, "path": frames[qid]["depth"]})
            yield (e["scene_id"], e["role"], key, geo["REPROJ7_HARMONIC"], geo["near7"].astype(bool), geo["cosine"], gt,
                   np.isfinite(gt) & (gt > 0), arrays)


def f7(arrays, arm, r7, cosine):
    alpha = arrays[f"{arm}_alpha"].astype(np.float64)
    ray = arrays[f"{arm}_depth_z"].astype(np.float64) / np.maximum(alpha, 1e-12) / cosine
    return np.where(alpha >= 0.5, ray, r7), alpha >= 0.5, ray


def gm(a, b):
    return np.sqrt(np.maximum(a, 1e-6) * np.maximum(b, 1e-6))


def readouts(r7, zs, jn, cosine):
    ttt, ttt_covered, ttt_ray = f7(zs, "TTT_m1", r7, cosine)
    jn_f7, jn_covered, _ = f7(jn, "CACHE", r7, cosine)
    return {"ZS_TTT_F7": ttt, "JN_F7": jn_f7, "CORE_B": gm(r7, ttt),
            "MASK": gm(r7, np.where(ttt_covered & jn_covered, ttt_ray, r7)),
            "TRIPLE": np.cbrt(np.maximum(r7, 1e-6) * np.maximum(ttt, 1e-6) * np.maximum(jn_f7, 1e-6)),
            "PAIR": gm(ttt, jn_f7)}


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
    if (root / "selection.json").exists() or (root / "audit/selection_locked_at.txt").exists() or (root / "scores").exists():
        raise SystemExit("a selection or a confirmation score exists already")
    rows, access, started = [], [], time.time()
    for cohort in ("ARKIT_SYSTEM", "ARKIT_SYSTEM2"):
        for scene, role, key, r7, near7, cosine, gt, valid, arrays in queries(root, cohort, access):
            values = {}
            for name, p in {"R7": r7, **readouts(r7, arrays["ZS"], arrays["JN1"], cosine)}.items():
                regional(values, name, p, gt, valid, near7)
            rows.append({"cohort": "DEV", "scene_id": scene, "role": role, "query": key, "values": values})
    table = scene_table(rows)
    means = {name: float(np.mean(list(table[f"DEV:{name}|ALL"].values())))
             for name in ("R7", "ZS_TTT_F7", "JN_F7", "CORE_B", *CANDIDATES)}
    best = min(CANDIDATES, key=lambda c: (means[c], CANDIDATES.index(c)))
    selection = {"dev_cohorts": ["ARKIT_SYSTEM", "ARKIT_SYSTEM2"], "dev_scenes": len(table["DEV:CORE_B|ALL"]),
                 "dev_queries": len(rows), "dev_means_all": means, "selected": best,
                 "stop": not means[best] < means["CORE_B"], "confirmation_query_depth_read": False,
                 "seconds": round(time.time() - started, 1)}
    write(root / "selection.json", selection)
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/selection_locked_at.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} "
                                                        f"sha256(selection.json)={sha(root / 'selection.json')}\n")
    print("SELECTED", best, "stop" if selection["stop"] else "confirm", json.dumps({k: round(v, 5) for k, v in means.items()}))


def score(root):
    stamp = (root / "audit/selection_locked_at.txt").read_text()
    if f"sha256(selection.json)={sha(root / 'selection.json')}" not in stamp:
        raise PermissionError("selection.json changed after the selection")
    selection = json.loads((root / "selection.json").read_text())
    if selection["stop"]:
        print("NO_CANDIDATE_ON_DEV: no confirmation query depth is read")
        return
    best = selection["selected"]
    rows, access, started = [], [], time.time()
    for cohort, seeds in (("ARKIT_SYSTEM3", ("JN1", "JN2", "JN3")), ("ARKIT_VAL", ("JN1",))):
        for scene, role, key, r7, near7, cosine, gt, valid, arrays in queries(root, cohort, access):
            values = {}
            regional(values, "R7", r7, gt, valid, near7)
            for seed in seeds:
                out = readouts(r7, arrays["ZS"], arrays[seed], cosine)
                if seed == "JN1":
                    for name in ("ZS_TTT_F7", "JN_F7", "CORE_B", *CANDIDATES):
                        regional(values, name, out[name], gt, valid, near7)
                    regional(values, "SYS", out[best], gt, valid, near7)
                else:
                    regional(values, f"SYS_{seed}", out[best], gt, valid, near7)
            rows.append({"cohort": cohort, "scene_id": scene, "role": role, "query": key, "values": values})
    table = scene_table(rows)
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access, "selected": best,
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries; selected", best)


S3, VAL = "ARKIT_SYSTEM3", "ARKIT_VAL"
PRIMARY = {"P1_JEPA_ABSTAIN_VS_CORE_B_SYSTEM3_ALL": (f"{S3}:CORE_B|ALL", f"{S3}:SYS|ALL"),
           "P2_JEPA_ABSTAIN_VS_CORE_B_VAL_ALL": (f"{VAL}:CORE_B|ALL", f"{VAL}:SYS|ALL")}
SECONDARY = {"S1_SEED_JN2_SYSTEM3_ALL": (f"{S3}:CORE_B|ALL", f"{S3}:SYS_JN2|ALL"),
             "S2_SEED_JN3_SYSTEM3_ALL": (f"{S3}:CORE_B|ALL", f"{S3}:SYS_JN3|ALL"),
             "S3_SYSTEM3_NEAR7": (f"{S3}:CORE_B|NEAR7", f"{S3}:SYS|NEAR7"),
             "S4_SYSTEM3_FAR7": (f"{S3}:CORE_B|FAR7", f"{S3}:SYS|FAR7"),
             "S5_VS_MODEL_FREE_SYSTEM3_ALL": (f"{S3}:R7|ALL", f"{S3}:SYS|ALL"),
             "S6_VS_TTT_ALONE_SYSTEM3_ALL": (f"{S3}:ZS_TTT_F7|ALL", f"{S3}:SYS|ALL"),
             "S7_VS_JN_ALONE_SYSTEM3_ALL": (f"{S3}:JN_F7|ALL", f"{S3}:SYS|ALL")}
NAMES = ("JEPA_ABSTAIN_IMPROVES_CORE_B", "JEPA_ABSTAIN_IMPROVES_CORE_B_PARTIAL", "JEPA_ABSTAIN_CORE_B_NOT_ESTABLISHED",
         "JEPA_ABSTAIN_HARMS_CORE_B")


def label(status):
    return {"SUPPORTED": NAMES[0], "PARTIAL": NAMES[1], "HARMFUL": NAMES[3]}.get(status, NAMES[2])


def analyze(root):
    selection = json.loads((root / "selection.json").read_text())
    if selection["stop"]:
        write(root / "results/CA_results.json", {"selection": selection, "branches": {"P1": "NO_CANDIDATE_ON_DEV",
                                                                                       "P2": "NO_CANDIDATE_ON_DEV"}})
        print("ANALYZED NO_CANDIDATE_ON_DEV", selection["selected"], json.dumps(selection["dev_means_all"]))
        return
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    branches = {"P1": label(primary["P1_JEPA_ABSTAIN_VS_CORE_B_SYSTEM3_ALL"]["status"]),
                "P2": label(primary["P2_JEPA_ABSTAIN_VS_CORE_B_VAL_ALL"]["status"])}
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    outputs = Path(__file__).resolve().parents[2]
    v3 = json.loads((outputs / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/results/ARKIT_SYSTEM3_results.json").read_text())["means"]
    replication = {"V3:CORE_B|ALL": abs(means[f"{S3}:CORE_B|ALL"]["mean"] - v3["CORE_B|ALL"]["mean"]),
                   "V3:ZS_TTT_F7|ALL": abs(means[f"{S3}:ZS_TTT_F7|ALL"]["mean"] - v3["ZS_TTT_F7|ALL"]["mean"])}
    write(root / "results/CA_results.json", {"selection": selection, "branches": branches, "primary": primary,
                                             "secondary": secondary, "means": means, "replication_abs_diff": replication})
    print("ANALYZED", branches, "selected", selection["selected"])
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} median {s['median']:+.4f} {s['status']}")
    for k, v in means.items():
        if k.endswith("|ALL"):
            print(f"  {k:28s} {v['mean']:.4f} ({v['scenes']})")
    print("  replication:", json.dumps({k: round(v, 12) for k, v in replication.items()}))


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"select": select, "score": score, "analyze": analyze}[stage](root)
