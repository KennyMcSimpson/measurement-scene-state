"""DESCRIPTIVE, post hoc (after EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1): does averaging G1 with harmonic interpolation
also help on Replica (8 scenes, a different data source)? Too few scenes for the V2 gate; reports scene means and
how many scenes improve. Stages (run through cv_host_run.py from the repository root, CUDA hidden):
  python replica_hg.py ROOT make      the root: Replica manifest copy + G1 (three seeds) arms contract
  (then: scripts/evaluate_rgbd_carrier_arms.py --root ROOT --stage seal --cohort REPLICA)
  python replica_hg.py ROOT score     H3, G1, HG = (H3 + G1) / 2 on FAR and ALL (NEAR from H3)
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
G1_ROOT = REPO / "outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/G1/EXP-3D-RGBD-LONG-TRAIN-V17"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
if stage == "make":
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE, root / "manifest_replica.json")
    v17 = json.loads((REPO / "outputs/EXP-3D-RGBD-LONG-TRAIN-V17/training_contract.json").read_text())["eval"]
    contract = {"arms": {"G1": {"module": "mcss.mechanism_pilot.rgbd_long_train_carrier", "root": str(G1_ROOT),
                                "step": 24000, "preregistration_sha256": sha(G1_ROOT / "preregistration.json")}},
                "reference_arm": "G1", "seeds": [20260928, 20260929, 20260930],
                "cohorts": {"REPLICA": {"manifest": str(root / "manifest_replica.json"),
                                        "manifest_sha256": sha(root / "manifest_replica.json"), "split": "REPLICA",
                                        "role": "primary_query"}},
                "near_px": v17["near_px"], "reprojection_fill_constant_m": v17["reprojection_fill_constant_m"],
                "abstain_opacity_threshold": v17["abstain_opacity_threshold"]}
    (root / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
    print("REPLICA_ROOT", root)
else:
    manifest = json.loads((root / "manifest_replica.json").read_text())
    seal = json.loads((root / "raw/eval/REPLICA/prediction_seal.json").read_text())
    assert seal["query_depth_read"] is False
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in manifest["scenes"]:
        frames = {f["frame_id"]: f for f in record["frames"]}
        for role in ("A", "B"):
            for qid in record["roles"]["primary_query"]:
                item = seal["predictions"][f"REPLICA/{record['scene_id']}/{role}/{qid}"]
                assert sha(item["path"]) == item["sha256"]
                p = dict(np.load(item["path"]))
                gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
                valid = np.isfinite(gt) & (gt > 0)
                near = p["near"].astype(bool)
                h3 = p["REPROJ_HARMONIC"]
                g1 = np.mean([p[f"G1_{s}"] for s in seal["seeds"]], axis=0)
                for name, pred in (("H3", h3), ("G1", g1), ("HG", (h3 + g1) / 2)):
                    for region, mask, full in (("FAR", valid & ~near, pred), ("ALL", valid, np.where(near, h3, pred))):
                        if mask.any():
                            rows[f"{name}|{region}"][record["scene_id"]][role].append(
                                float(np.mean(np.abs(full[mask] - gt[mask]) / gt[mask])))
    table = {m: {s: float(np.mean([np.mean(v) for v in r.values()])) for s, r in sc.items()} for m, sc in rows.items()}
    out = {"note": "descriptive, post hoc; 8 Replica scenes; no gate", "means": {}, "scenes_better_than_H3": {}}
    for key, values in sorted(table.items()):
        out["means"][key] = float(np.mean(list(values.values())))
    for name in ("G1", "HG"):
        for region in ("FAR", "ALL"):
            h, x = table[f"H3|{region}"], table[f"{name}|{region}"]
            common = sorted(set(h) & set(x))
            out["scenes_better_than_H3"][f"{name}|{region}"] = [sum(h[s] > x[s] for s in common), len(common)]
    out["per_scene"] = table
    (root / "replica_hg_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_scene"}, indent=1))
