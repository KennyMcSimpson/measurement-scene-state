"""EXP-3D-CAJU3D-FUSENET-V1 descriptive analysis after the confirmation (no decision depends on it):
  per-seed place means of every FuseNet on both confirmation cohorts;
  the AbsRel of covered (alpha >= 0.5) and uncovered pixels, and the uncovered split by near7, for CORE_B / KF_BASE /
  FUSE_BASE (ensemble);
  the uncovered correction r of FUSE_BASE: mean, share of |r| > 0.6 (near the 0.7 bound), sign;
  the per-place gain CORE_B - FUSE_BASE against the place's CORE_B error (Spearman) and its quartiles.
Run from the repository root with the DepthSplat environment, CUDA hidden (PYTHONPATH=src):
  python fn_describe.py ROOT > ROOT/results/describe.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "code"))
import fn_data as D  # noqa: E402
import fn_model as M  # noqa: E402
import fn_score as FS  # noqa: E402

K, S = FS.K, D.S


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def main():
    tables = json.loads(FS.TABLES.read_text())
    models = FS.load_models(ROOT)
    out = {}
    for cohort in FS.CONF:
        access = []
        queries = D.load_cohort(ROOT, cohort, access, with_jepa=True)
        per_seed_rows, rows = [], []
        acc = {k: [0.0, 0] for k in ("cov_CORE_B", "cov_KF_BASE", "cov_FUSE_BASE", "unc_CORE_B", "unc_FUSE_BASE",
                                     "uncnear_CORE_B", "uncnear_FUSE_BASE", "uncfar_CORE_B", "uncfar_FUSE_BASE")}
        r_all = []
        for q in queries:
            valid = q["gt"] > 0
            gt = q["gt"][valid]
            base = torch.tensor(q["base"][None])
            jepa = torch.tensor(q["jepa"][None])
            seed_preds = {}
            with torch.no_grad():
                for arm, ms in models.items():
                    for i, m in enumerate(ms):
                        logit, r = m(base, jepa if m.use_jepa else None)
                        seed_preds[f"{arm}_{i}"] = np.exp(M.log_prediction(base, logit, r)[0].double().numpy())[valid]
                        if arm == "BASE":
                            r_all.append(r[0].numpy()[(q["base"][6] < 0.5) & valid])
                fuse = np.exp(FS.ensemble_log(models["BASE"], base, jepa, True)[0].double().numpy())[valid]
            per_seed_rows.append({"place": q["place"], "episode": q["episode"], "query": q["key"],
                                  "core_b": {k: float(np.mean(np.abs(v - gt) / gt)) for k, v in seed_preds.items()}})
            core = K.readout(q["kf"], "CORE_B", tables)
            kfb = K.readout(q["kf"], "KF_BASE", tables)
            cov = q["kf"]["alpha"] >= 0.5
            near = q["kf"]["near"]
            for name, pred in (("CORE_B", core), ("KF_BASE", kfb), ("FUSE_BASE", fuse)):
                err = np.abs(pred - gt) / gt
                for key, mask in (("cov", cov), ("unc", ~cov), ("uncnear", ~cov & near), ("uncfar", ~cov & ~near)):
                    if f"{key}_{name}" in acc:
                        acc[f"{key}_{name}"][0] += float(err[mask].sum())
                        acc[f"{key}_{name}"][1] += int(mask.sum())
            rows.append({"place": q["place"], "episode": q["episode"], "query": q["key"],
                         "core_b": {"CORE_B": float(np.mean(np.abs(core - gt) / gt)),
                                    "FUSE_BASE": float(np.mean(np.abs(fuse - gt) / gt))}})
        seed_table = S.place_table(per_seed_rows)
        table = S.place_table(rows)
        places = sorted(table["CORE_B"])
        core_err = np.array([table["CORE_B"][p] for p in places])
        gain = np.array([table["CORE_B"][p] - table["FUSE_BASE"][p] for p in places])
        r_cat = np.concatenate(r_all)
        total_pixels = sum(v[1] for k, v in acc.items() if k.startswith("cov_CORE_B") or k.startswith("unc_CORE_B"))
        out[cohort] = {
            "per_seed_means": {k: round(S.mean(v), 5) for k, v in seed_table.items()},
            "pixel_absrel": {k: round(v[0] / max(v[1], 1), 5) for k, v in acc.items()},
            "uncovered_pixel_share": round(acc["unc_CORE_B"][1] / max(total_pixels, 1), 4),
            "uncovered_far_share_of_uncovered": round(acc["uncfar_CORE_B"][1] / max(acc["unc_CORE_B"][1], 1), 4),
            "r": {"mean": round(float(r_cat.mean()), 4), "share_positive": round(float((r_cat > 0).mean()), 4),
                  "share_abs_gt_0.6": round(float((np.abs(r_cat) > 0.6).mean()), 4)},
            "gain_vs_core_b_error_spearman": round(float(np.corrcoef(rank(core_err), rank(gain))[0, 1]), 3),
            "gain_quartiles": [round(float(x), 5) for x in np.quantile(gain, [0, 0.25, 0.5, 0.75, 1])],
            "worse_places": int((gain < 0).sum()), "places": len(places),
        }
        print(cohort, json.dumps(out[cohort], indent=1))
    (ROOT / "results/describe.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
