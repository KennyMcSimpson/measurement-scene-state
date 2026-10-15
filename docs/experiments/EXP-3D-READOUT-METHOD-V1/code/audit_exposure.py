"""EXP-3D-READOUT-METHOD-V1 Phase 0: data-exposure and model-lineage audit from metadata only (no image, depth,
prediction array or ground truth is read). Writes results/exposure_audit.json.
Run from the repository root:  python audit_exposure.py OUT_DIR
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("outputs")
DATA = {  # data cohort -> (data experiment, manifest)
    "ARKIT_PILOT_TRAINVAL": ("EXP-3D-RGBD-ARKIT-DATA-V1", "manifest_arkit_trainval.json"),
    "ARKIT_CONFIRM": ("EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1", None),
    "ARKIT_SELECT": ("EXP-3D-RGBD-ARKIT-SELECT-DATA-V1", None),
    "ARKIT_SYSTEM": ("EXP-3D-RGBD-ARKIT-SYSTEM-DATA-V1", "manifest_arkit_system.json"),
    "ARKIT_SYSTEM2": ("EXP-3D-RGBD-ARKIT-SYSTEM2-DATA-V1", "manifest_arkit_system2.json"),
    "ARKIT_SYSTEM3": ("EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1", "manifest_arkit_system3.json"),
    "ARKIT_SYSTEM4": ("EXP-3D-RGBD-ARKIT-SYSTEM4-DATA-V1", "manifest_arkit_system4.json"),
    "ARKIT_VAL": ("EXP-3D-RGBD-ARKIT-VAL-DATA-V1", None),
}


def load(path):
    return json.loads(Path(path).read_text())


def manifest_of(exp, name):
    root = OUT / exp
    if name is None:
        cands = sorted(root.glob("manifest_*.json"))
        if not cands:
            return None, []
        name = cands[0].name
    m = load(root / name)
    return str(root / name), m["scenes"]


def block_tail(exp):
    """Downloaded-but-never-prepared scans of a data cohort's selection block (metadata only)."""
    root = OUT / exp
    try:
        sel = load(root / "selection.json")
    except FileNotFoundError:
        return None
    block = sel.get("block") or [s for fold in sel.get("selected", {}).values() for s in fold]
    if not block:
        return None
    prepared = set()
    for mf in root.glob("manifest_*.json"):
        prepared |= {r["scene_id"].removeprefix("arkit_") for r in load(mf)["scenes"]}
    failures = set(load(root / "scene_failures.json")) if (root / "scene_failures.json").exists() else set()
    downloads = load(root / "downloads.json").get("scans", {}) if (root / "downloads.json").exists() else {}
    done = prepared | failures
    tail = [s["video_id"] for s in block if s["video_id"] not in done]
    return {"block": len(block), "prepared_eligible": len(prepared), "failures": len(failures),
            "never_prepared": len(tail), "never_prepared_downloaded": sum(v in downloads for v in tail),
            "downloads_recorded": len(downloads)}


def main():
    out_dir = Path(sys.argv[1])
    report = {"cohorts": {}, "overlaps": {}, "models": {}, "zero_hit_queries": {}, "block_tails": {}}
    places, scans = {}, {}
    for name, (exp, mname) in DATA.items():
        path, scenes = manifest_of(exp, mname)
        if path is None:
            report["cohorts"][name] = {"manifest": None}
            continue
        splits = Counter(r.get("split") for r in scenes)
        report["cohorts"][name] = {"manifest": path, "scans": len(scenes), "splits": dict(splits)}
        for split in splits:
            key = name if len(splits) == 1 else f"{name}:{split}"
            places[key] = {r["physical_scene_id"] for r in scenes if r.get("split") == split}
            scans[key] = {r["scene_id"] for r in scenes if r.get("split") == split}
        report["block_tails"][name] = block_tail(exp)
    keys = sorted(places)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            n = len(places[a] & places[b])
            if n:
                report["overlaps"][f"{a} & {b}"] = n
    # evaluation cohorts as episode sets (frame-role locks)
    locks = {
        "KFUSE root": OUT / "EXP-3D-CAJU3D-KFUSE-V1/frame_role_lock.json",
        "CAJU-V1 root": OUT / "EXP-3D-CAJU-HEADROOM-V1/frame_role_lock.json",
        "SYSTEM4 root": OUT / "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1/frame_role_lock.json",
        "ENGINE/JEPAGATE DEV root": OUT / "EXP-3D-CAJU3D-ENGINE-V1/frame_role_lock.json",
    }
    eval_places = {}
    for root_name, path in locks.items():
        for cohort, entry in load(path)["cohorts"].items():
            recs = {r["scene_id"]: r for r in load(entry["manifest"])["scenes"]}
            eps = entry["episodes"]
            pl = {e.get("place") or recs[e["scene_id"]]["physical_scene_id"] for e in eps}
            eval_places[f"{root_name}:{cohort}"] = pl
            report["cohorts"].setdefault("episode_sets", {})[f"{root_name}:{cohort}"] = {
                "episodes": len(eps), "places": len(pl), "queries": sum(len(e["queries"]) for e in eps)}
    # models and the places they trained / validated on
    pilot = load(OUT / "EXP-3D-RGBD-ARKIT-DATA-V1/manifest_arkit_trainval.json")["scenes"]
    report["models"]["DepthSplat released (re10k+dl3dv)"] = {"arkit_training_places": 0}
    report["models"]["H-JEPA (JEPAGATE-V1)"] = {
        "train_places": len({r["physical_scene_id"] for r in pilot if r["split"] == "TRAIN"}),
        "monitor_places": len({r["physical_scene_id"] for r in pilot if r["split"] == "DEV"})}
    kf = load(OUT / "EXP-3D-CAJU3D-KFUSE-V1/frame_role_lock.json")["cohorts"]
    fit_places = {e["place"] for c in ("FIT_SYSTEM", "FIT_SYSTEM2") for e in kf[c]["episodes"]}
    report["models"]["KF_BASE table (KFUSE-V1)"] = {"fit_places": len(fit_places), "cohorts": ["FIT_SYSTEM", "FIT_SYSTEM2"]}
    fn = load(OUT / "EXP-3D-CAJU3D-FUSENET-V1/results/train_log.json")
    report["models"]["FUSE V1 (FUSENET-V1 BASE/JEPA)"] = {"train_places": len(fn["train_places"]),
                                                           "val_places": len(fn["val_places"]),
                                                           "subset_of_fit": set(fn["train_places"] + fn["val_places"]) <= fit_places}
    v2 = load(OUT / "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1/results/v2_train_log.json")
    report["models"]["FUSE V2 (SYSTEM4-HOLDOUT-V1)"] = {"places": v2["places"], "train_places": v2["train_places"],
                                                        "val_places": len(v2["val_places"]),
                                                        "cohorts": ["FIT_SYSTEM", "FIT_SYSTEM2", "CONF_SYSTEM3",
                                                                    "CONF_VAL", "CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL"]}
    # exposure of each evaluation place set to each model's training
    hjepa_train = {r["physical_scene_id"] for r in pilot if r["split"] in ("TRAIN", "DEV")}
    v2_places = fit_places | eval_places["KFUSE root:CONF_SYSTEM3"] | eval_places["KFUSE root:CONF_VAL"] | \
        set().union(*(eval_places[f"CAJU-V1 root:{c}"] for c in ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL")))
    for name, pl in eval_places.items():
        report["cohorts"]["episode_sets"][name].update({
            "in_hjepa_training": len(pl & hjepa_train), "in_kf_fit": len(pl & fit_places),
            "in_fuse_v1_training": len(pl & set(fn["train_places"] + fn["val_places"])),
            "in_fuse_v2_training": len(pl & v2_places)})
    # zero-hit queries from the geometry seals (hit7 = 0: the whole R7 is the Hypersim TRAIN constant)
    seals = {"FIT_SYSTEM": "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1/geometry/geometry_seal_ARKIT_SYSTEM.json",
             "FIT_SYSTEM2": "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2/geometry/geometry_seal_ARKIT_SYSTEM2.json",
             "CONF_SYSTEM3": "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/geometry/geometry_seal_ARKIT_SYSTEM3.json",
             "CONF_VAL": "EXP-3D-RGBD-ENSEMBLES-ARKIT-V1/geometry/geometry_seal_ARKIT_VAL.json",
             "ARKIT_SYSTEM4": "EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1/geometry/geometry_seal_ARKIT_SYSTEM4.json"}
    for c in ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL"):
        seals[c] = f"EXP-3D-CAJU-HEADROOM-V1/geometry/geometry_seal_{c}.json"
    for name, rel in seals.items():
        preds = load(OUT / rel)["predictions"]
        q = {k: v for k, v in preds.items() if v.get("kind", "query") == "query"}
        hits = [v["hit7"] for v in q.values()]
        report["zero_hit_queries"][name] = {"queries": len(hits), "zero_hit": sum(h == 0 for h in hits),
                                            "hit7_below_0.1": sum(h < 0.1 for h in hits),
                                            "median_hit7": sorted(hits)[len(hits) // 2]}
    report["fill_constant"] = {"value_m": load(OUT / "EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/lock.json")["constant_m"],
                               "source": "EXP-3D-RGBD-DEPTH-BOUNDS-V7 decision_rules GEOMETRY_FREE_REFERENCE "
                                         "REF_TRAIN_ABSREL_OPTIMAL (Hypersim TRAIN scenes ai_*)",
                               "used_when": "a query has no reprojection hit at all (harmonic_fill)"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exposure_audit.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=list) + "\n")
    print(json.dumps({k: report[k] for k in ("overlaps", "models", "zero_hit_queries", "block_tails")}, indent=1,
                     default=list))
    print(json.dumps(report["cohorts"]["episode_sets"], indent=1))


if __name__ == "__main__":
    main()
