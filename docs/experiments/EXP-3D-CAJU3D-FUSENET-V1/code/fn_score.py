"""EXP-3D-CAJU3D-FUSENET-V1: the preregistered confirmation (CONF_SYSTEM3 = P1, CONF_VAL = P2).

Arms (per query, valid ground-truth pixels; place = mean over episodes of the mean over queries):
  CORE_B      Core B (float64, kf_logic.readout)
  KF_BASE     KFUSE-V1's frozen table (float64, kf_logic.readout; sha256 checked)
  FUSE_BASE   mean log prediction of the three BASE FuseNets
  FUSE_JEPA   mean log prediction of the three JEPA FuseNets
  FUSE_BASE_W / FUSE_JEPA_W   the same weights with the uncovered correction switched off (R7 there)
P = FUSE_BASE - FUSE_JEPA (> 0: the H-JEPA's prediction improves the learned readout), V1's paired place bootstrap +
V2 gate; secondaries KF_BASE - FUSE_BASE, CORE_B - FUSE_BASE, CORE_B - FUSE_JEPA, FUSE_BASE_W - FUSE_BASE,
KF_BASE - FUSE_BASE_W. Models are checked against audit/fit_log.txt before any confirmation truth is read.
Run from the repository root with the DepthSplat environment (PYTHONPATH=src):  python fn_score.py ROOT
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OUTPUTS / "EXP-3D-CAJU3D-KFUSE-V1" / "code"))
import fn_data as D  # noqa: E402
import fn_model as M  # noqa: E402
import kf_logic as K  # noqa: E402
from score_analyze import contrast  # noqa: E402

S = D.S
CONF = ("CONF_SYSTEM3", "CONF_VAL")
TABLES = OUTPUTS / "EXP-3D-CAJU3D-KFUSE-V1" / "fusion_tables.json"
TABLES_SHA = "15100550ca6b39763244a69405ec88f8eede4faaa700aff4619557ee1c7e533c"
NAMES = ("CORE_B", "KF_BASE", "FUSE_BASE", "FUSE_JEPA", "FUSE_BASE_W", "FUSE_JEPA_W")
SECONDARY = (("KF_BASE", "FUSE_BASE"), ("CORE_B", "FUSE_BASE"), ("CORE_B", "FUSE_JEPA"), ("FUSE_BASE_W", "FUSE_BASE"),
             ("KF_BASE", "FUSE_BASE_W"), ("KF_BASE", "FUSE_JEPA"))


def branch(p1, p2):
    if p1["status"] == "SUPPORTED" and p2["mean"] > 0:
        return "JEPA_FUSION_SUPPORTED"
    if p1["status"] in ("SUPPORTED", "PARTIAL"):
        return "JEPA_FUSION_PARTIAL"
    if p1["status"] == "HARMFUL":
        return "JEPA_FUSION_HARMFUL"
    return "JEPA_FUSION_NOT_ESTABLISHED"


def load_models(root):
    logged = {line.split()[-2]: line.split()[-1] for line in (root / "audit/fit_log.txt").read_text().splitlines()}
    models = {"BASE": [], "JEPA": []}
    for name, digest in sorted(logged.items()):
        path = root / "models" / f"{name}.pt"
        if D.sha(path) != digest:
            raise PermissionError(f"{path}: changed since training")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = M.FuseNet(ckpt["use_jepa"])
        model.load_state_dict(ckpt["state_dict"])
        models[ckpt["arm"]].append(model.eval())
    if any(len(v) != 3 for v in models.values()):
        raise PermissionError("expected three models per arm")
    return models


@torch.no_grad()
def ensemble_log(models, base, jepa, use_residual):
    outs = [M.log_prediction(base, *m(base, jepa if m.use_jepa else None), use_residual=use_residual) for m in models]
    return torch.stack(outs).mean(dim=0)


def main():
    root = Path(sys.argv[1]).resolve()
    if D.sha(TABLES) != TABLES_SHA:
        raise PermissionError("the KFUSE-V1 fusion tables changed")
    tables = json.loads(TABLES.read_text())
    models = load_models(root)
    results = {}
    for cohort in CONF:
        access = []
        queries = D.load_cohort(root, cohort, access, with_jepa=True)
        rows = []
        for q in queries:
            valid = q["gt"] > 0
            kq = q["kf"]
            preds = {"CORE_B": K.readout(kq, "CORE_B", tables), "KF_BASE": K.readout(kq, "KF_BASE", tables)}
            base = torch.tensor(q["base"][None])
            jepa = torch.tensor(q["jepa"][None])
            for arm in ("BASE", "JEPA"):
                for suffix, residual in (("", True), ("_W", False)):
                    log_pred = ensemble_log(models[arm], base, jepa, residual)[0].double().numpy()
                    preds[f"FUSE_{arm}{suffix}"] = np.exp(log_pred)[valid]
            rows.append({"place": q["place"], "episode": q["episode"], "query": q["key"],
                         "core_b": {n: float(np.mean(np.abs(preds[n] - kq["gt"]) / kq["gt"])) for n in NAMES}})
        table = S.place_table(rows)
        results[cohort] = {"places": len(table["CORE_B"]), "queries": len(rows), "ground_truth_reads": len(access),
                           "means": {n: S.mean(table[n]) for n in NAMES},
                           "primary_base_minus_jepa": contrast(table, "FUSE_BASE", "FUSE_JEPA"),
                           "secondary": {f"{l}_minus_{r}": contrast(table, l, r) for l, r in SECONDARY}}
        (root / f"scores/{cohort}").mkdir(parents=True, exist_ok=True)
        (root / f"scores/{cohort}/rows.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    results["branch"] = branch(results["CONF_SYSTEM3"]["primary_base_minus_jepa"],
                               results["CONF_VAL"]["primary_base_minus_jepa"])
    (root / "results").mkdir(exist_ok=True)
    (root / "results/confirm_results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    print("CONFIRM", results["branch"], json.dumps({c: {n: round(v, 5) for n, v in results[c]["means"].items()}
                                                    for c in CONF}))


if __name__ == "__main__":
    main()
