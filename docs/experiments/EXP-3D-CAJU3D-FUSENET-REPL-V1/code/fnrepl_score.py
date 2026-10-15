"""EXP-3D-CAJU3D-FUSENET-REPL-V1: the three frozen FUSE_BASE networks of FUSENET-V1 on the 58 CAJU-V1 places (CPU).

Inputs = fn_data.base_image from CAJU-V1's FULL replay (released weights, 7-view CACHE, 100 TTT steps) and its
geometry seals; ground truth through TeacherAccess. Arms CORE_B and KF_BASE (KFUSE-V1's frozen table, float64),
FUSE_BASE (mean log prediction of the three BASE models) and FUSE_BASE_W (uncovered correction off).
P = CORE_B - FUSE_BASE over the 58 places (V1 paired place bootstrap + V2 gate).
Run from the repository root with the DepthSplat environment, CUDA hidden (PYTHONPATH=src):  python fnrepl_score.py ROOT
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
FN = OUTPUTS / "EXP-3D-CAJU3D-FUSENET-V1"
CAJU = OUTPUTS / "EXP-3D-CAJU-HEADROOM-V1"
sys.path.insert(0, str(FN / "code"))
import fn_data as D  # noqa: E402
import fn_score as FS  # noqa: E402
from caju_logic import core_b  # noqa: E402
from score_analyze import contrast  # noqa: E402

K, S = FS.K, D.S
COHORTS = ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL")
NAMES = ("CORE_B", "KF_BASE", "FUSE_BASE", "FUSE_BASE_W")
VERDICT = {"SUPPORTED": "REPLICATED", "PARTIAL": "PARTIALLY_REPLICATED", "HARMFUL": "REVERSED"}


def replay_queries():
    out = {}
    for part in sorted((CAJU / "replay").glob("*/replay_seal_*of*.json")):
        seal = json.loads(part.read_text())
        if seal["ttt_steps"] != 100 or not seal["checkpoint_sha256"].startswith("f8ddd845"):
            raise PermissionError(f"{part}: not the released-weights 100-step replay")
        out.update({k: v for k, v in seal["predictions"].items() if v["kind"] == "query"})
    return out


def main():
    root = Path(sys.argv[1]).resolve()
    if not (root / "audit/locked_at.txt").is_file():
        raise SystemExit("lock first")
    if D.sha(FS.TABLES) != FS.TABLES_SHA:
        raise PermissionError("the KFUSE-V1 fusion tables changed")
    tables = json.loads(FS.TABLES.read_text())
    models = FS.load_models(FN)["BASE"]
    locks = json.loads((CAJU / "frame_role_lock.json").read_text())["cohorts"]
    replay = replay_queries()
    access, rows = [], []
    pixels = {k: [0.0, 0] for k in ("cov_CORE_B", "cov_FUSE_BASE", "unc_CORE_B", "unc_FUSE_BASE")}
    for cohort in COHORTS:
        lock = locks[cohort]
        if D.sha(lock["manifest"]) != lock["manifest_sha256"]:
            raise PermissionError(f"{cohort}: manifest changed")
        geometry = json.loads((CAJU / "geometry" / f"geometry_seal_{cohort}.json").read_text())["predictions"]
        records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
        for e in lock["episodes"]:
            teacher = S.TeacherAccess(records[e["scene_id"]], e["queries"], S.IMAGE_SIZE, S.REPO, access)
            for qid in e["queries"]:
                key = f"{e['scene_id']}/{e['role']}/{qid}"
                ttt, geo = S.load_checked(replay[key]), S.load_checked(geometry[key])
                base = D.base_image(ttt["FULL_depth_z"], ttt["FULL_alpha"], geo["REPROJ7_HARMONIC"], geo["near7"],
                                    geo["cosine"])
                gt = teacher.query_depth(qid)
                valid = np.isfinite(gt) & (gt > 0)
                _, f7 = core_b(ttt["FULL_depth_z"], ttt["FULL_alpha"], geo["REPROJ7_HARMONIC"], geo["cosine"])
                kq = {"gt": gt[valid], "r7": np.maximum(geo["REPROJ7_HARMONIC"][valid].astype(np.float64), 1e-6),
                      "f7": np.maximum(f7[valid], 1e-6), "alpha": ttt["FULL_alpha"][valid].astype(np.float64),
                      "near": geo["near7"][valid].astype(bool), "log_p": np.zeros(int(valid.sum()))}
                b = torch.tensor(base[None])
                preds = {"CORE_B": K.readout(kq, "CORE_B", tables), "KF_BASE": K.readout(kq, "KF_BASE", tables),
                         "FUSE_BASE": np.exp(FS.ensemble_log(models, b, None, True)[0].double().numpy())[valid],
                         "FUSE_BASE_W": np.exp(FS.ensemble_log(models, b, None, False)[0].double().numpy())[valid]}
                cov = kq["alpha"] >= 0.5
                for name in ("CORE_B", "FUSE_BASE"):
                    err = np.abs(preds[name] - kq["gt"]) / kq["gt"]
                    for tag, mask in (("cov", cov), ("unc", ~cov)):
                        pixels[f"{tag}_{name}"][0] += float(err[mask].sum())
                        pixels[f"{tag}_{name}"][1] += int(mask.sum())
                rows.append({"cohort": cohort, "place": e["place"], "episode": f"{e['scene_id']}/{e['role']}",
                             "query": key, "core_b": {n: float(np.mean(np.abs(preds[n] - kq["gt"]) / kq["gt"]))
                                                      for n in NAMES}})
    if {a["purpose"] for a in access} != {"SEALED_QUERY_EVALUATION"}:
        raise PermissionError("unexpected read purposes")
    table = S.place_table(rows)
    places = sorted(table["CORE_B"])
    gain = np.array([table["CORE_B"][p] - table["FUSE_BASE"][p] for p in places])
    primary = contrast(table, "CORE_B", "FUSE_BASE")
    result = {"places": len(places), "queries": len(rows), "ground_truth_reads": len(access),
              "means": {n: S.mean(table[n]) for n in NAMES}, "primary_core_b_minus_fuse_base": primary,
              "secondary": {"KF_BASE_minus_FUSE_BASE": contrast(table, "KF_BASE", "FUSE_BASE"),
                            "FUSE_BASE_W_minus_FUSE_BASE": contrast(table, "FUSE_BASE_W", "FUSE_BASE")},
              "pixel_absrel": {k: v[0] / max(v[1], 1) for k, v in pixels.items()},
              "gain_quartiles": np.quantile(gain, [0, 0.25, 0.5, 0.75, 1]).tolist(), "worse_places": int((gain < 0).sum()),
              "by_cohort": {c: {n: S.mean(S.place_table([r for r in rows if r["cohort"] == c])[n]) for n in NAMES}
                            for c in COHORTS},
              "verdict": VERDICT.get(primary["status"], "NOT_REPLICATED")}
    (root / "results").mkdir(exist_ok=True)
    (root / "results/repl_results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (root / "scores").mkdir(exist_ok=True)
    (root / "scores/rows.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print("FN_REPL", result["verdict"], json.dumps({n: round(v, 5) for n, v in result["means"].items()}))


if __name__ == "__main__":
    main()
