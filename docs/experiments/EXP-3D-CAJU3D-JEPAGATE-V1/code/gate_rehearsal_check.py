"""EXP-3D-CAJU3D-JEPAGATE-V1 rehearsal check (before the lock; no ground truth read): the NONE arm against the
sealed Core B (ZS TTT_m1) of the same queries, as a Core B readout relative-difference distribution, next to each
gate arm's distance from NONE. Run from the repository root (CUDA hidden, PYTHONPATH=src)."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs")
for p in ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code", "EXP-3D-CAJU-HEADROOM-V1/code", "EXP-3D-CAJU3D-ENGINE-V1/code"):
    sys.path.insert(0, str((OUT / p).resolve()))
import engine_score as S  # noqa: E402
from caju_logic import core_b  # noqa: E402

E = OUT / "EXP-3D-CAJU3D-JEPAGATE-V1"
seal = json.loads((Path.home() / "cv_work/caju3d/gate_rehearsal/gate_seal_0of1.json").read_text())
reused = json.loads((E / "reused_seals.json").read_text())["DEV_SYSTEM"]
zs = S.reused_predictions(reused["ZS"])
geo = json.loads(Path(reused["geometry"]["path"]).read_text())["predictions"]


def rel(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-6)


def q(x):
    return [float(np.quantile(x, t)) for t in (0.5, 0.9, 0.99)] + [float(x.max()), float(x.mean())]


pool = {"NONE_vs_SEALED": [], "SEALED_alpha_absdiff": []}
for key, item in seal["predictions"].items():
    new = S.load_checked(item)
    old = S.load_checked(zs[key])
    g = S.load_checked(geo[key])
    cb_old, _ = core_b(old["TTT_m1_depth_z"], old["TTT_m1_alpha"], g["REPROJ7_HARMONIC"], g["cosine"])
    cb = {arm: core_b(new[f"{arm}_depth_z"], new[f"{arm}_alpha"], g["REPROJ7_HARMONIC"], g["cosine"])[0]
          for arm in ("NONE", "JEPA", "DEPTH", "INC", "RANDOM")}
    pool["NONE_vs_SEALED"].append(rel(cb["NONE"], cb_old).ravel())
    pool["SEALED_alpha_absdiff"].append(np.abs(new["NONE_alpha"] - old["TTT_m1_alpha"]).ravel())
    for arm in ("JEPA", "DEPTH", "INC", "RANDOM"):
        pool.setdefault(f"{arm}_vs_NONE", []).append(rel(cb[arm], cb["NONE"]).ravel())
    cov_new = float((new["NONE_alpha"] >= 0.5).mean())
    cov_old = float((old["TTT_m1_alpha"] >= 0.5).mean())
    print(key, "coverage NONE", round(cov_new, 4), "sealed", round(cov_old, 4),
          "max |NONE-sealed| depth_z", float(np.abs(new["NONE_depth_z"] - old["TTT_m1_depth_z"]).max()))
print("quantiles: p50 p90 p99 max mean")
for name, parts in pool.items():
    print(f"{name:22s}", [f"{v:.2e}" for v in q(np.concatenate(parts))])
