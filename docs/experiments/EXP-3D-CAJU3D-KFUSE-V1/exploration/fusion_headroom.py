"""CAJU-3D step 3 feasibility probe (exploratory, DEV only; no GPU): where Core B's error sits and how much a
per-pixel fusion of the TTT readout (F7) and the 7-view reprojection (R7) could gain, cross-validated over places.

DEV = the 24 places / 48 episodes of ENGINE-V1 and JEPAGATE-V1 (their truth has been read already). Uses the sealed
released-weights TTT_m1 (Core B's TTT), the sealed geometry (R7, cosine) and the DEV ground truth through
TeacherAccess. Per valid query pixel:
  obs   alpha >= 0.5: F7 is the TTT ray depth;  hole  alpha < 0.5: F7 falls back to R7 (Core B = R7)
  fused prediction R7^(1 - w) F7^w (w = 0.5 is Core B)
Reports: the AbsRel share and level of obs / hole; the per-pixel oracle min(F7, R7) on obs; the best uniform w; and
binned weights w(bin) of simple features fitted on the other places (leave-one-place-out), for
  DIS  |log F7 - log R7|,  ALPHA  the TTT alpha,  DEPTH  log R7.
Run from the repository root (CUDA hidden, PYTHONPATH=src):  python fusion_headroom.py OUT.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("outputs").resolve()
for _path in ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code", "EXP-3D-CAJU-HEADROOM-V1/code", "EXP-3D-CAJU3D-ENGINE-V1/code"):
    sys.path.insert(0, str(OUT / _path))
import engine_score as S  # noqa: E402
from caju_logic import core_b  # noqa: E402

ROOT = OUT / "EXP-3D-CAJU3D-JEPAGATE-V1"
COHORTS = ("DEV_SYSTEM", "DEV_SYSTEM2")
WEIGHTS = np.round(np.linspace(0.0, 1.0, 21), 2)
BINS = 8


def main():
    locks = json.loads((ROOT / "frame_role_lock.json").read_text())["cohorts"]
    reused = json.loads((ROOT / "reused_seals.json").read_text())
    access, pixels = [], defaultdict(list)  # place -> list of per-query dicts
    for cohort in COHORTS:
        lock = locks[cohort]
        if S.sha(lock["manifest"]) != lock["manifest_sha256"]:
            raise PermissionError(f"{cohort}: manifest changed")
        geo_entry = reused[cohort]["geometry"]
        if S.sha(geo_entry["path"]) != geo_entry["sha256"]:
            raise PermissionError(f"{cohort}: geometry seal changed")
        geometry = json.loads(Path(geo_entry["path"]).read_text())["predictions"]
        zs = S.reused_predictions(reused[cohort]["ZS"])
        records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
        for e in lock["episodes"]:
            teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
            for qid in e["queries"]:
                key = f"{e['scene_id']}/{e['role']}/{qid}"
                geo, ttt = S.load_checked(geometry[key]), S.load_checked(zs[key])
                gt = teacher.query_depth(qid)
                valid = np.isfinite(gt) & (gt > 0)
                cb, f7 = core_b(ttt["TTT_m1_depth_z"], ttt["TTT_m1_alpha"], geo["REPROJ7_HARMONIC"], geo["cosine"])
                r7 = np.maximum(geo["REPROJ7_HARMONIC"].astype(np.float64), 1e-6)
                f7 = np.maximum(f7, 1e-6)
                alpha = ttt["TTT_m1_alpha"].astype(np.float64)
                pixels[e["place"]].append({"episode": f"{e['scene_id']}/{e['role']}", "gt": gt[valid],
                                           "r7": r7[valid], "f7": f7[valid], "alpha": alpha[valid],
                                           "cb": cb[valid]})
    if {a["purpose"] for a in access} != {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")

    def place_mean(per_query_values):
        by_episode = defaultdict(list)
        for episode, v in per_query_values:
            by_episode[episode].append(v)
        return float(np.mean([np.mean(v) for v in by_episode.values()]))

    def absrel(pred, gt):
        return np.abs(pred - gt) / gt

    places = sorted(pixels)
    result = {"places": len(places), "queries": sum(len(v) for v in pixels.values()), "ground_truth_reads": len(access)}
    # 1. where the error sits (Core B)
    share = defaultdict(list)
    for p in places:
        for q in pixels[p]:
            obs = q["alpha"] >= 0.5
            e = absrel(q["cb"], q["gt"])
            share["obs_pixel_share"].append(obs.mean())
            share["obs_error_share"].append(e[obs].sum() / max(e.sum(), 1e-12))
            share["obs_absrel"].append(e[obs].mean() if obs.any() else np.nan)
            share["hole_absrel"].append(e[~obs].mean() if (~obs).any() else np.nan)
            share["obs_f7_absrel"].append(absrel(q["f7"][obs], q["gt"][obs]).mean() if obs.any() else np.nan)
            share["obs_r7_absrel"].append(absrel(q["r7"][obs], q["gt"][obs]).mean() if obs.any() else np.nan)
            share["obs_oracle_absrel"].append(np.minimum(absrel(q["f7"][obs], q["gt"][obs]),
                                                         absrel(q["r7"][obs], q["gt"][obs])).mean()
                                              if obs.any() else np.nan)
    result["where"] = {k: float(np.nanmean(v)) for k, v in share.items()}

    # 2. place means of Core B, F7-only, R7-only, the per-pixel oracle and the uniform weights
    def table(fn):
        return {p: place_mean([(q["episode"], fn(q)) for q in pixels[p]]) for p in places}

    # per query: the obs pixels' AbsRel under every weight (E [len(WEIGHTS), n_obs]) and the holes' AbsRel sum
    for p in places:
        for q in pixels[p]:
            obs = q["alpha"] >= 0.5
            lr, lf, g = np.log(q["r7"][obs]), np.log(q["f7"][obs]), q["gt"][obs]
            q["obs"], q["n"] = obs, obs.size
            q["E"] = np.stack([np.abs(np.exp((1 - w) * lr + w * lf) - g) / g for w in WEIGHTS])
            q["hole_sum"] = float(absrel(q["r7"][~obs], q["gt"][~obs]).sum())

    tables = {"CORE_B": table(lambda q: absrel(q["cb"], q["gt"]).mean()),
              "F7": table(lambda q: absrel(q["f7"], q["gt"]).mean()),
              "R7": table(lambda q: absrel(q["r7"], q["gt"]).mean()),
              "ORACLE_PIXEL": table(lambda q: (q["E"].min(axis=0).sum() + q["hole_sum"]) / q["n"])}
    for j, w in enumerate(WEIGHTS):
        tables[f"W{w:.2f}"] = table(lambda q, j=j: (q["E"][j].sum() + q["hole_sum"]) / q["n"])

    # 3. leave-one-place-out: the uniform w and binned w(feature) chosen on the other places
    feats = {"DIS": lambda q: np.abs(np.log(q["f7"]) - np.log(q["r7"]))[q["obs"]],
             "ALPHA": lambda q: q["alpha"][q["obs"]], "DEPTH": lambda q: np.log(q["r7"])[q["obs"]]}
    loo = {"LOO_UNIFORM": {}}
    loo.update({f"LOO_{f}": {} for f in feats})
    for held in places:
        train = [p for p in places if p != held]
        best_w = min(WEIGHTS, key=lambda w: np.mean([tables[f"W{w:.2f}"][p] for p in train]))
        loo["LOO_UNIFORM"][held] = tables[f"W{best_w:.2f}"][held]
        for name, f in feats.items():
            edges = np.quantile(np.concatenate([f(q) for p in train for q in pixels[p]]),
                                np.linspace(0, 1, BINS + 1)[1:-1])
            # per bin: the w minimizing the summed AbsRel of the training obs pixels in the bin
            loss = np.zeros((BINS, len(WEIGHTS)))
            for p in train:
                for q in pixels[p]:
                    b = np.digitize(f(q), edges)
                    for j in range(len(WEIGHTS)):
                        loss[:, j] += np.bincount(b, weights=q["E"][j], minlength=BINS)
            j_bin = np.argmin(loss, axis=1)

            def apply(q, f=f, edges=edges, j_bin=j_bin):
                j = j_bin[np.digitize(f(q), edges)]
                return (q["E"][j, np.arange(j.size)].sum() + q["hole_sum"]) / q["n"]

            loo[f"LOO_{name}"][held] = place_mean([(q["episode"], apply(q)) for q in pixels[held]])
            loo.setdefault(f"w_bins_{name}", []).append(WEIGHTS[j_bin].tolist())
    tables.update({k: v for k, v in loo.items() if k.startswith("LOO_")})
    means = {k: float(np.mean(list(v.values()))) for k, v in tables.items()}
    result["means"] = means
    result["vs_core_b"] = {k: {"mean": float(np.mean([tables["CORE_B"][p] - v[p] for p in places])),
                               "better_places": int(sum(tables["CORE_B"][p] - v[p] > 0 for p in places))}
                           for k, v in tables.items() if k != "CORE_B"}
    result["w_bins_example"] = {k: v[0] for k, v in loo.items() if k.startswith("w_bins_")}
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["where"], indent=1))
    for k in ("CORE_B", "F7", "R7", "ORACLE_PIXEL", "W0.50", "W0.70", "W0.90", "W1.00", "LOO_UNIFORM", "LOO_DIS",
              "LOO_ALPHA", "LOO_DEPTH"):
        v = result["vs_core_b"].get(k, {"mean": 0.0, "better_places": 0})
        print(f"{k:14s} {means[k]:.5f}  vs Core B {v['mean']:+.5f}  better at {v['better_places']}/{len(places)}")
    print("w bins (first fold):", json.dumps(result["w_bins_example"]))


if __name__ == "__main__":
    main()
