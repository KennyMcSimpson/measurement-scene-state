"""EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1: sealing every readout on ARKIT_SYSTEM4 before any SYSTEM4 truth is read (CPU).

Per query, from this root's sealed geometry (REPROJ7_HARMONIC, near7, cosine) and released-weights TTT (ds_eval TTT_m1):
CORE_B (float64), KF_BASE (KFUSE-V1's frozen table), FUSE_V1 (FUSENET-V1's three BASE networks, hash-checked) and FUSE_V2
(this root's three V2 networks, hash-checked against audit/v2_fit_log.txt), saved as one npz per query with a seal.
Run from the repository root with the DepthSplat environment, CUDA hidden (PYTHONPATH=src):  python s4_readouts.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s4_v2 as V  # noqa: E402  (sets the import paths)
import fn_score as FS  # noqa: E402

D, K, S = V.D, V.K, V.S
COHORT = "ARKIT_SYSTEM4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def v2_models(root):
    models = []
    for line in (root / "audit/v2_fit_log.txt").read_text().splitlines():
        name, digest = line.split()[-2:]
        path = root / "models_v2" / f"{name}.pt"
        if sha(path) != digest:
            raise PermissionError(f"{path}: changed since training")
        model = V.V2Net()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        models.append(model.eval())
    if len(models) != 3:
        raise PermissionError("expected three V2 networks")
    return models


def main():
    root = Path(sys.argv[1]).resolve()
    out = root / "readouts"
    if (out / "readout_seal.json").exists():
        raise SystemExit("readouts are sealed once")
    if sha(V.TABLES) != V.TABLES_SHA:
        raise PermissionError("the KFUSE-V1 fusion tables changed")
    tables = json.loads(V.TABLES.read_text())
    v1 = FS.load_models(Path(V.OUTPUTS / "EXP-3D-CAJU3D-FUSENET-V1"))["BASE"]
    v2 = v2_models(root)
    lock = json.loads((root / "frame_role_lock.json").read_text())["cohorts"][COHORT]
    geo_seal = json.loads((root / "geometry" / f"geometry_seal_{COHORT}.json").read_text())
    if geo_seal["query_depth_read"] is not False:
        raise PermissionError("geometry seal read query depth")
    geometry = geo_seal["predictions"]
    ttt = {}
    parts = sorted((root / "ds" / COHORT / "ZS").glob("ds_seal_*of*.json"))
    if len(parts) != 4:
        raise PermissionError("incomplete ds seals")
    for part in parts:
        seal = json.loads(part.read_text())
        if seal["query_depth_read"] is not False or seal["scale_max"] != 0.1 or seal["limit"]:
            raise PermissionError(f"{part}: not the protocol's run")
        ttt.update(seal["predictions"])
    out.mkdir(parents=True)
    entries = {}
    for e in lock["episodes"]:
        for qid in e["queries"]:
            key = f"{e['scene_id']}/{e['role']}/{qid}"
            g, t = S.load_checked(geometry[key]), S.load_checked(ttt[key])
            depth_z, alpha = t["TTT_m1_depth_z"], t["TTT_m1_alpha"]
            r7, near7, cosine = g["REPROJ7_HARMONIC"], g["near7"], g["cosine"]
            cb, f7 = V.core_b(depth_z, alpha, r7, cosine)
            q = {"r7": np.maximum(r7.astype(np.float64), 1e-6).ravel(), "f7": np.maximum(f7, 1e-6).ravel(),
                 "alpha": alpha.astype(np.float64).ravel(), "near": near7.astype(bool).ravel(),
                 "log_p": np.zeros(r7.size)}
            base = torch.tensor(D.base_image(depth_z, alpha, r7, near7, cosine)[None])
            w_kf = torch.tensor(V.kf_weight_map(depth_z, alpha, r7, near7, cosine, tables)[None])
            with torch.no_grad():
                fuse_v1 = FS.ensemble_log(v1, base, None, True)[0].double().numpy()
                fuse_v2 = torch.stack([V.log_prediction(base, w_kf, m(base)) for m in v2]).mean(0)[0].double().numpy()
            path = out / f"{e['scene_id']}_{e['role']}_{qid}.npz"
            np.savez_compressed(path, CORE_B=cb, KF_BASE=K.readout(q, "KF_BASE", tables).reshape(r7.shape),
                                FUSE_V1=np.exp(fuse_v1), FUSE_V2=np.exp(fuse_v2), covered=(alpha >= 0.5))
            entries[key] = {"path": str(path), "sha256": sha(path)}
    (out / "readout_seal.json").write_text(json.dumps({
        "cohort": COHORT, "queries": len(entries), "query_depth_read": False, "fusion_tables_sha256": V.TABLES_SHA,
        "fuse_v1_models": (V.OUTPUTS / "EXP-3D-CAJU3D-FUSENET-V1/audit/fit_log.txt").read_text(),
        "fuse_v2_models": (root / "audit/v2_fit_log.txt").read_text(), "predictions": entries}, indent=1))
    print("S4_READOUTS_SEALED", len(entries))


if __name__ == "__main__":
    main()
