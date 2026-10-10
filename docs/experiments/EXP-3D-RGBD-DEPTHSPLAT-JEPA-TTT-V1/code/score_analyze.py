"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1: scoring, TTT step-size selection (DEV) and the preregistered analysis.

score:  every prediction of the cohort must be sealed and unchanged (arms seal: REPROJ_HARMONIC and G1 for three
        seeds; geometry seal: REPROJ7_HARMONIC, NEAR, NEAR7, ray cosine; one DepthSplat seal per state builder);
        only then is query depth read. Readouts per query:
          REPROJ_HARMONIC, REPROJ7_HARMONIC, G1 (per seed), FAR_G1 = NEAR ? REPROJ_HARMONIC : G1 (per seed);
          <M>_CTX_F    = alpha >= 0.5 ? DepthSplat ray depth : REPROJ_HARMONIC   (three-frame state);
          <M>_CTX_G1   = NEAR ? REPROJ_HARMONIC : (alpha >= 0.5 ? DepthSplat : G1)  (per G1 seed, averaged);
          <M>_<A>_F7   = alpha >= 0.5 ? DepthSplat ray depth : REPROJ7_HARMONIC  (A = CACHE or TTT_m<m>);
          <M>_<A>_H7   = NEAR7 ? REPROJ7_HARMONIC : <M>_<A>_F7  (measurement where it reaches, the state elsewhere);
        DepthSplat ray depth = (depth_z / alpha) / cosine. AbsRel on ALL (valid ground truth), NEAR, FAR (NEAR of
        the three context frames, 8 px) and NEAR7, FAR7 (of the seven frames); coverage = share of the region's
        pixels with alpha >= 0.5. Scene value = mean over roles of the mean over queries (empty regions skipped).
select: (DEV) the TTT step-size multiplier m with the lowest mean DEV ALL AbsRel of ZS_TTT_m<m>_F7 (ties: smaller m).
analyze: the preregistered contrasts with the project's paired scene bootstrap (10,000 draws, seed 20260928) and
        V2 gate; multi-seed builders (J1..J3) are averaged per scene first; branches for Core A and Core B.
Run through cv_host_run.py from the repository root (CUDA hidden):
  python score_analyze.py ROOT score COHORT --arms-seal SEAL --ds ZS=DIR [ZS3=DIR J1=DIR ...]
  python score_analyze.py ROOT select DEV
  python score_analyze.py ROOT analyze COHORT --ttt-mult M
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.direct_capacity_statistics import paired
from mcss.mechanism_pilot.geometry_carrier_statistics import _evidence, surface_status

ALPHA_MIN = 0.5
REGIONS = ("ALL", "NEAR", "FAR", "NEAR7", "FAR7")
DRAWS, SEED = 10000, 20260928
J_SEEDS = ("J1", "J2", "J3")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


def load_checked(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"a sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def ds_seal(directory, cohort, keys):
    """Merge a state builder's shard seals; every query key exactly once, one setting across shards."""
    parts = sorted(Path(directory).glob("ds_seal_*of*.json"))
    if not parts:
        raise FileNotFoundError(f"no DepthSplat seal in {directory}")
    predictions, settings = {}, None
    for part in parts:
        seal = json.loads(part.read_text())
        if seal["cohort"] != cohort or seal["query_depth_read"] is not False or seal["limit"]:
            raise PermissionError(f"{part}: wrong cohort, query depth read, or a rehearsal limit")
        setting = {k: seal[k] for k in ("model", "checkpoint_sha256", "scale_max", "arms", "ttt_mult", "ttt_steps")}
        if settings is None:
            settings = setting
        elif setting != settings:
            raise ValueError(f"{part}: shard settings differ")
        for key, item in seal["predictions"].items():
            if key in predictions:
                raise ValueError(f"{key} sealed twice in {directory}")
            predictions[key] = item
    if set(predictions) != set(keys):
        raise ValueError(f"{directory}: sealed queries differ from the frame-role lock")
    return predictions, settings, [str(p) for p in parts]


def absrel(prediction, gt, mask):
    return float(np.mean(np.abs(prediction[mask] - gt[mask]) / gt[mask])) if mask.any() else None


def score(args):
    root = args.root.resolve()
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][args.cohort]
    if sha(lock["manifest"]) != lock["manifest_sha256"]:
        raise PermissionError("manifest changed since the frame-role lock")
    records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
    keys = [f"{e['scene_id']}/{e['role']}/{q}" for e in lock["episodes"] for q in e["queries"]]
    arms = json.loads(args.arms_seal.read_text())
    if arms["query_depth_read"] is not False:
        raise PermissionError("arms seal does not certify that no query depth was read")
    geometry_path = root / "geometry" / f"geometry_seal_{args.cohort}.json"
    geometry = json.loads(geometry_path.read_text())
    if not geometry["replay_vs_arms_seal"]["pass"] or set(geometry["predictions"]) != set(keys):
        raise PermissionError("geometry seal incomplete or its replay failed")
    if geometry["arms_seal_sha256"] != sha(args.arms_seal):
        raise PermissionError("geometry seal was made against another arms seal")
    models = {}
    for spec in args.ds:
        name, directory = spec.split("=", 1)
        models[name] = ds_seal(directory, args.cohort, keys)
    for key in keys:  # completeness and integrity before any query depth is opened
        if f"{arms['cohort']}/{key}" not in arms["predictions"]:
            raise ValueError(f"arms seal misses {key}")
    access, rows = [], []
    started = time.time()
    for e in lock["episodes"]:
        frames = {f["frame_id"]: f for f in records[e["scene_id"]]["frames"]}
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            sealed = load_checked(arms["predictions"][f"{arms['cohort']}/{key}"])
            geo = load_checked(geometry["predictions"][key])
            ds = {name: load_checked(m[0][key]) for name, m in models.items()}
            gt = np.load(frames[qid]["depth"]).squeeze().astype(np.float64)
            access.append({"key": key, "path": frames[qid]["depth"]})
            valid = np.isfinite(gt) & (gt > 0)
            near, near7 = geo["near"].astype(bool), geo["near7"].astype(bool)
            masks = {"ALL": valid, "NEAR": valid & near, "FAR": valid & ~near, "NEAR7": valid & near7,
                     "FAR7": valid & ~near7}
            h3, h7, cosine = geo["REPROJ_HARMONIC"], geo["REPROJ7_HARMONIC"], geo["cosine"]
            g1 = [sealed[f"G1_{s}"] for s in arms["seeds"]]
            readouts = {"REPROJ_HARMONIC": [h3], "REPROJ7_HARMONIC": [h7], "G1": g1,
                        "FAR_G1": [np.where(near, h3, g) for g in g1]}
            coverage = {}
            for name, arrays in ds.items():
                arm_names = sorted({k.rsplit("_", 2)[0] for k in arrays if k.endswith("_depth_z")})
                for arm in arm_names:
                    alpha = arrays[f"{arm}_alpha"].astype(np.float64)
                    depth = arrays[f"{arm}_depth_z"].astype(np.float64)
                    covered = alpha >= ALPHA_MIN
                    ray = np.where(covered, depth / np.maximum(alpha, 1e-12) / cosine, np.nan)
                    if arm == "CTX":
                        readouts[f"{name}_CTX_F"] = [np.where(covered, ray, h3)]
                        readouts[f"{name}_CTX_G1"] = [np.where(near, h3, np.where(covered, ray, g)) for g in g1]
                    else:
                        readouts[f"{name}_{arm}_F7"] = [np.where(covered, ray, h7)]
                        readouts[f"{name}_{arm}_H7"] = [np.where(near7, h7, np.where(covered, ray, h7))]
                    for region, mask in masks.items():
                        coverage[f"{name}_{arm}_cov_{region}"] = float(covered[mask].mean()) if mask.any() else None
            values = {}
            for readout, predictions in readouts.items():
                for region, mask in masks.items():
                    per_seed = [absrel(p, gt, mask) for p in predictions]
                    values[f"{readout}|{region}"] = None if per_seed[0] is None else float(np.mean(per_seed))
            for name, value in coverage.items():
                values[f"{name}|COVERAGE"] = value
            rows.append({"scene_id": e["scene_id"], "role": e["role"], "query": qid, "values": values})
    table = scene_table(rows)
    out = root / "scores" / args.cohort
    write(out / "query_rows.json", rows)
    write(out / "scene_table.json", table)
    write(out / "score_info.json", {
        "cohort": args.cohort, "queries": len(rows), "arms_seal": str(args.arms_seal),
        "arms_seal_sha256": sha(args.arms_seal), "geometry_seal_sha256": sha(geometry_path),
        "ds": {name: {"settings": m[1], "seals": m[2], "seal_sha256": [sha(p) for p in m[2]]}
               for name, m in models.items()},
        "query_depth_reads": access, "seconds": round(time.time() - started, 1)})
    print("SCORED", args.cohort, len(rows), "queries", len(table), "readout-regions")


def scene_table(rows):
    """{readout|region: {scene: value}}: per role the mean over queries, per scene the mean over roles; seed groups
    J_* = mean of J1..J3 per scene (scenes present for all three)."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, value in row["values"].items():
            if value is not None and np.isfinite(value):
                grouped[name][row["scene_id"]][row["role"]].append(value)
    table = {name: {scene: float(np.mean([np.mean(v) for v in roles.values()])) for scene, roles in scenes.items()}
             for name, scenes in grouped.items()}
    rests = {name[len(s):] for name in table for s in J_SEEDS if name.startswith(f"{s}_")}
    for rest in sorted(rests):  # the seeds that completed (at least two, by the preregistered failure rule)
        parts = [table[f"{s}{rest}"] for s in J_SEEDS if f"{s}{rest}" in table]
        if len(parts) >= 2:
            common = set.intersection(*(set(p) for p in parts))
            table[f"J{rest}"] = {s: float(np.mean([p[s] for p in parts])) for s in sorted(common)}
    return table


def contrast(table, left, right):
    """Paired scene contrast left - right (> 0: right has the lower error), with the V2 gate."""
    a, b = table[left], table[right]
    common = sorted(set(a) & set(b))
    stats = paired({s: a[s] for s in common}, {s: b[s] for s in common}, draws=DRAWS, seed=SEED)
    checks = {
        "mean_ci_and_scene_consistency": _evidence(stats),
        "every_leave_one_scene_out_gain_positive": bool(stats["loso"]) and min(stats["loso"].values()) > 0,
        "positive_top1_share_at_most_0_5": stats["positive_top1_contribution"] is not None
        and stats["positive_top1_contribution"] <= 0.5,
    }
    status = surface_status(stats, checks)
    stats.pop("per_scene", None)
    stats.pop("loso", None)
    return {"left": left, "right": right, "scenes": len(common), "status": status, "checks": checks, **stats}


def select(args):
    table = json.loads((args.root.resolve() / "scores" / args.cohort / "scene_table.json").read_text())
    names = sorted(k for k in table if k.startswith("ZS_TTT_m") and k.endswith("_F7|ALL"))
    means = {k: float(np.mean(list(table[k].values()))) for k in names}
    mult = lambda k: float(k[len("ZS_TTT_m"):].split("_")[0])  # noqa: E731
    best = min(names, key=lambda k: (round(means[k], 12), mult(k)))
    result = {"cohort": args.cohort, "criterion": "lowest mean DEV ALL AbsRel of ZS_TTT_m<m>_F7; ties: smaller m",
              "means": means, "reference_means": {k: float(np.mean(list(table[k].values()))) for k in
                                                  ("ZS_CACHE_F7|ALL", "REPROJ7_HARMONIC|ALL") if k in table},
              "selected_mult": mult(best)}
    write(args.root.resolve() / "ttt_selection.json", result)
    print("TTT_SELECTED", result["selected_mult"], json.dumps(means))


PRIMARY = {
    "A1_JEPA_VS_HARMONIC_FAR": ("REPROJ_HARMONIC|FAR", "J_CTX_F|FAR"),
    "B1_TTT_BEYOND_CACHE_ALL": ("ZS_CACHE_F7|ALL", "ZS_TTT_{m}_F7|ALL"),
    "B2_TTT_VS_REPROJ7_ALL": ("REPROJ7_HARMONIC|ALL", "ZS_TTT_{m}_F7|ALL"),
}
SECONDARY = {
    "A2a_JEPA_TRAINING_GAIN_FAR_SAME_CAP": ("ZS3_CTX_F|FAR", "J_CTX_F|FAR"),
    "A2b_JEPA_TRAINING_GAIN_FAR_VS_RELEASED": ("ZS_CTX_F|FAR", "J_CTX_F|FAR"),
    "A3_JEPA_OVER_G1_FAR": ("FAR_G1|FAR", "J_CTX_G1|FAR"),
    "A4a_JEPA_VS_HARMONIC_ALL": ("REPROJ_HARMONIC|ALL", "J_CTX_F|ALL"),
    "A4b_JEPA_VS_HARMONIC_NEAR": ("REPROJ_HARMONIC|NEAR", "J_CTX_F|NEAR"),
    "A5_ZS_VS_HARMONIC_FAR": ("REPROJ_HARMONIC|FAR", "ZS_CTX_F|FAR"),
    "A6_G1_HYBRID_VS_HARMONIC_FAR": ("REPROJ_HARMONIC|FAR", "FAR_G1|FAR"),
    "B3_TTT_BEYOND_CACHE_FAR7": ("ZS_CACHE_F7|FAR7", "ZS_TTT_{m}_F7|FAR7"),
    "B4_STREAM_CACHE_GAIN_ALL": ("ZS_CTX_F|ALL", "ZS_CACHE_F7|ALL"),
    "B5_REPROJ7_GAIN_ALL": ("REPROJ_HARMONIC|ALL", "REPROJ7_HARMONIC|ALL"),
    "B6_CACHE_HYBRID_VS_REPROJ7_ALL": ("REPROJ7_HARMONIC|ALL", "ZS_CACHE_H7|ALL"),
    "B7_JEPA_CACHE_HYBRID_VS_REPROJ7_ALL": ("REPROJ7_HARMONIC|ALL", "J_CACHE_H7|ALL"),
    "B8_TTT_HYBRID_VS_REPROJ7_ALL": ("REPROJ7_HARMONIC|ALL", "ZS_TTT_{m}_H7|ALL"),
    "B9_TTT_VS_REPROJ7_FAR7": ("REPROJ7_HARMONIC|FAR7", "ZS_TTT_{m}_F7|FAR7"),
}
MEANS = ("REPROJ_HARMONIC", "REPROJ7_HARMONIC", "G1", "FAR_G1", "ZS_CTX_F", "ZS3_CTX_F", "J_CTX_F", "J_CTX_G1",
         "ZS_CTX_G1", "ZS_CACHE_F7", "ZS_TTT_{m}_F7", "ZS_CACHE_H7", "ZS_TTT_{m}_H7", "J_CACHE_F7", "J_CACHE_H7",
         "J1_CTX_F", "J2_CTX_F", "J3_CTX_F")
COVERAGE = ("ZS_CTX_cov", "ZS3_CTX_cov", "J_CTX_cov", "ZS_CACHE_cov", "ZS_TTT_{m}_cov", "J_CACHE_cov")


def branch_a(status):
    return {"SUPPORTED": "A_SUPPORTED", "PARTIAL": "A_PARTIAL", "HARMFUL": "A_HARMFUL"}.get(status,
                                                                                          "A_NOT_ESTABLISHED")


def branch_b(b1, b2):
    if b1 == "SUPPORTED":
        return "B_SUPPORTED_PRACTICAL" if b2 == "SUPPORTED" else "B_MECHANISM_ONLY"
    return {"PARTIAL": "B_PARTIAL", "HARMFUL": "B_HARMFUL"}.get(b1, "B_NOT_ESTABLISHED")


def analyze(args):
    root = args.root.resolve()
    table = json.loads((root / "scores" / args.cohort / "scene_table.json").read_text())
    m = f"m{args.ttt_mult:g}"

    def fill(name):
        return name.replace("{m}", m)

    def run(spec):
        out = {}
        for label, (left, right) in spec.items():
            left, right = fill(left), fill(right)
            out[label] = contrast(table, left, right) if left in table and right in table else {
                "left": left, "right": right, "status": "NO_DATA"}
        return out

    primary, secondary = run(PRIMARY), run(SECONDARY)
    means = {}
    for name in MEANS:
        for region in REGIONS:
            key = f"{fill(name)}|{region}"
            if key in table:
                means[key] = {"mean": float(np.mean(list(table[key].values()))), "scenes": len(table[key])}
    for name in COVERAGE:  # J_* coverage is the per-scene mean over J1..J3 (scene_table)
        for region in REGIONS:
            key = f"{fill(name)}_{region}|COVERAGE"
            if key in table:
                means[key] = {"mean": float(np.mean(list(table[key].values()))), "scenes": len(table[key])}
    result = {
        "cohort": args.cohort, "ttt_mult": args.ttt_mult,
        "core_a_branch": branch_a(primary["A1_JEPA_VS_HARMONIC_FAR"]["status"]),
        "core_b_branch": branch_b(primary["B1_TTT_BEYOND_CACHE_ALL"]["status"],
                                  primary["B2_TTT_VS_REPROJ7_ALL"]["status"]),
        "primary": primary, "secondary": secondary, "means": means,
        "statistics": {"unit": "scene", "draws": DRAWS, "seed": SEED, "gate": "V2 (mean > 0, CI95 low > 0, >= 75% "
                       "scenes not worse, every leave-one-scene-out mean > 0, positive top-1 share <= 0.5)"},
    }
    write(root / "results" / f"{args.cohort}_results.json", result)
    print("ANALYZED", args.cohort, result["core_a_branch"], result["core_b_branch"])
    for label, stats in {**primary, **secondary}.items():
        if stats["status"] != "NO_DATA":
            print(f"  {label}: {stats['mean']:+.4f} [{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}] "
                  f"{stats['positive']}/{stats['tied']}/{stats['negative']} {stats['status']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("score", "select", "analyze"))
    parser.add_argument("cohort")
    parser.add_argument("--arms-seal", type=Path)
    parser.add_argument("--ds", nargs="+", default=[])
    parser.add_argument("--ttt-mult", type=float)
    args = parser.parse_args()
    {"score": score, "select": select, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
