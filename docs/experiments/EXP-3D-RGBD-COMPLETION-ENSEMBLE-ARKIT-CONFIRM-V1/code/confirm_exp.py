"""EXP-3D-RGBD-COMPLETION-ENSEMBLE-ARKIT-CONFIRM-V1: the normalized-G1 completion ensemble on fresh real scans.

Stages (repository root, through cv_host_run.py, CUDA hidden):
  make    root: manifest copy (EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1, 80 scans), G1 arms contract (three seeds),
          frame-role lock (record x role; context = role frames; queries = primary_query). No prediction, no query depth.
  (then:  scripts/evaluate_rgbd_carrier_arms.py --root ROOT --stage seal --cohort ARKIT_CONFIRM)
  score   after the seal only: per query on FAR (valid pixels outside the 8-px NEAR of the context reprojection):
          H3 = REPROJ_HARMONIC; G1 = mean of the three seeds' sealed depth; G1N = per seed depth / opacity where
          opacity >= 0.5, else H3, then the mean over seeds; HGN = (H3 + G1N) / 2; HG = (H3 + G1) / 2; H3x0.5 = 0.5 H3.
          ALL versions use H3 in NEAR. Scene value = mean over roles of the mean over queries.
  analyze V1's paired scene bootstrap (10,000 draws, seed 20260928) and V2 gate.
  python confirm_exp.py ROOT make | score | analyze
"""
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import contrast, write  # noqa: E402

COHORT = "ARKIT_CONFIRM"
REPO = Path.cwd()
SOURCE = REPO / "outputs/EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1/manifest_arkit_confirm.json"
G1_ROOT = REPO / "outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/G1/EXP-3D-RGBD-LONG-TRAIN-V17"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make(root):
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "manifest_arkit_confirm.json"
    shutil.copyfile(SOURCE, manifest_path)
    records = json.loads(manifest_path.read_text())["scenes"]
    if len(records) != 80 or any(r["split"] != COHORT for r in records):
        raise ValueError("unexpected confirmation manifest")
    used = set()
    for prior in ("outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1/manifest_arkit_val80.json",):
        used |= {r["physical_scene_id"] for r in json.loads((REPO / prior).read_text())["scenes"]}
    if {r["physical_scene_id"] for r in records} & used:
        raise ValueError("a confirmation venue was used before")
    v17 = json.loads((REPO / "outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json").read_text())["eval"]
    contract = {
        "arms": {"G1": {"module": "mcss.mechanism_pilot.rgbd_long_train_carrier", "root": str(G1_ROOT), "step": 24000,
                        "preregistration_sha256": sha(G1_ROOT / "preregistration.json")}},
        "reference_arm": "G1", "seeds": [20260928, 20260929, 20260930],
        "cohorts": {COHORT: {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "split": COHORT,
                             "role": "primary_query"}},
        "near_px": v17["near_px"], "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
        "abstain_opacity_threshold": v17["abstain_opacity_threshold"]}
    (root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
    episodes = [{"scene_id": r["scene_id"], "role": role, "context": list(r["roles"][key]),
                 "queries": list(r["roles"]["primary_query"])}
                for r in records for role, key in (("A", "context_a"), ("B", "context_b"))]
    (root / "frame_role_lock.json").write_text(json.dumps({"cohorts": {COHORT: {
        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "episodes": episodes}}}, indent=1))
    print("CONFIRM_ROOT", root, len(records), "records", len(episodes), "episodes")


def score(root):
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    seal = json.loads((root / f"raw/eval/{COHORT}/prediction_seal.json").read_text())
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    if seal["query_depth_read"] is not False or any(f"{COHORT}/{k}" not in seal["predictions"] for k in keys):
        raise PermissionError("arms seal incomplete or not certified")
    rows, access, started = [], [], time.time()
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            item = seal["predictions"][f"{COHORT}/{e['scene_id']}/{e['role']}/{qid}"]
            if sha(item["path"]) != item["sha256"]:
                raise PermissionError("a sealed prediction changed")
            p = dict(np.load(item["path"]))
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": f"{e['scene_id']}/{e['role']}/{qid}", "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            near = p["near"].astype(bool)
            far = valid & ~near
            h3 = p["REPROJ_HARMONIC"]
            g1 = np.mean([p[f"G1_{s}"] for s in seal["seeds"]], axis=0)
            g1n = np.mean([np.where(p[f"OPACITY_G1_{s}"] >= 0.5,
                                    p[f"G1_{s}"] / np.maximum(p[f"OPACITY_G1_{s}"].astype(np.float64), 1e-6), h3)
                           for s in seal["seeds"]], axis=0)
            preds = {"H3": h3, "G1": g1, "G1N": g1n, "HGN": (h3 + g1n) / 2, "HG": (h3 + g1) / 2, "H3x0.5": 0.5 * h3}
            values = {"G1_abstain_far": float(np.mean([np.mean(p[f"OPACITY_G1_{s}"][far] < 0.5)
                                                       for s in seal["seeds"]])) if far.any() else None}
            for name, x in preds.items():
                values[f"{name}|FAR"] = float(np.mean(np.abs(x[far] - gt[far]) / gt[far])) if far.any() else None
                full = np.where(near, h3, x)
                values[f"{name}|ALL"] = float(np.mean(np.abs(full[valid] - gt[valid]) / gt[valid]))
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "values": values})
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, v in row["values"].items():
            if v is not None and np.isfinite(v):
                grouped[name][row["scene_id"]][row["role"]].append(v)
    table = {n: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for n, sc in grouped.items()}
    write(root / "scores/query_rows.json", rows)
    write(root / "scores/scene_table.json", table)
    write(root / "scores/score_info.json", {"queries": len(rows), "query_depth_reads": access,
                                            "arms_seal_sha256": sha(root / f"raw/eval/{COHORT}/prediction_seal.json"),
                                            "seconds": round(time.time() - started, 1)})
    print("SCORED", len(rows), "queries")


PRIMARY = {"C1_NORMALIZED_ENSEMBLE_VS_HARMONIC_FAR": ("H3|FAR", "HGN|FAR")}
SECONDARY = {"S1_G1N_VS_HARMONIC_FAR": ("H3|FAR", "G1N|FAR"), "S2_RAW_HG_VS_HARMONIC_FAR": ("H3|FAR", "HG|FAR"),
             "S3_SHRINK_ONLY_VS_HARMONIC_FAR": ("H3|FAR", "H3x0.5|FAR"),
             "S4_HGN_VS_SHRINK_ONLY_FAR": ("H3x0.5|FAR", "HGN|FAR"),
             "S5_NORMALIZED_ENSEMBLE_VS_HARMONIC_ALL": ("H3|ALL", "HGN|ALL"),
             "S6_RAW_G1_VS_HARMONIC_FAR": ("H3|FAR", "G1|FAR")}


def analyze(root):
    table = json.loads((root / "scores/scene_table.json").read_text())
    primary = {k: contrast(table, a, b) for k, (a, b) in PRIMARY.items()}
    secondary = {k: contrast(table, a, b) for k, (a, b) in SECONDARY.items()}
    status = primary["C1_NORMALIZED_ENSEMBLE_VS_HARMONIC_FAR"]["status"]
    branch = {"SUPPORTED": "NORMALIZED_ENSEMBLE_CONSISTENT", "PARTIAL": "NORMALIZED_ENSEMBLE_PARTIAL",
              "HARMFUL": "NORMALIZED_ENSEMBLE_HARMFUL"}.get(status, "NORMALIZED_ENSEMBLE_NOT_ESTABLISHED")
    means = {k: {"mean": float(np.mean(list(v.values()))), "scenes": len(v)} for k, v in sorted(table.items())}
    write(root / "results/ARKIT_CONFIRM_results.json", {"cohort": COHORT, "branch": branch, "primary": primary,
                                                        "secondary": secondary, "means": means})
    print("ANALYZED", branch)
    for name, s in {**primary, **secondary}.items():
        print(f"  {name}: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} {s['status']}")
    for k, v in means.items():
        print(f"  {k:16s} {v['mean']:.4f} ({v['scenes']})")


if __name__ == "__main__":
    root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    {"make": make, "score": score, "analyze": analyze}[stage](root)
