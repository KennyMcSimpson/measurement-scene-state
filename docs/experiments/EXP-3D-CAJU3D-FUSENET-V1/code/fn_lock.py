"""EXP-3D-CAJU3D-FUSENET-V1: the preregistration lock (before the query features are computed, before any training
and before any confirmation truth is read). Hashes the protocol, the code, the root files and the imported files;
keeps a copy of the locked protocol. Refuses when features, models, scores or results exist.
Run through cv_host_run.py from the repository root (CUDA hidden):  python fn_lock.py ROOT
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_model as M  # noqa: E402
import fn_train as T  # noqa: E402

OUT = Path("outputs").resolve()
IMPORTED = ("EXP-3D-CAJU3D-JEPAGATE-V1/model/hjepa_final.pth", "EXP-3D-CAJU3D-JEPAGATE-V1/code/hjepa.py",
            "EXP-3D-CAJU3D-KFUSE-V1/code/kf_jepa_core.py", "EXP-3D-CAJU3D-KFUSE-V1/code/kf_query_jepa.py",
            "EXP-3D-CAJU3D-KFUSE-V1/code/kf_logic.py", "EXP-3D-CAJU3D-KFUSE-V1/fusion_tables.json",
            "EXP-3D-CAJU3D-ENGINE-V1/code/engine_score.py", "EXP-3D-CAJU-HEADROOM-V1/code/caju_logic.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/score_analyze.py")
ROOT_FILES = ("PROTOCOL.md", "frame_role_lock.json", "reused_seals.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    for name in ("features", "models", "scores", "results", "preregistration.json"):
        if (root / name).exists():
            raise SystemExit(f"cannot lock: {name} exists already")
    prereg = {
        "experiment": "EXP-3D-CAJU3D-FUSENET-V1", "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root_files": {n: sha(root / n) for n in ROOT_FILES},
        "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.suffix in (".py", ".sh")},
        "imported": {rel: sha(OUT / rel) for rel in IMPORTED},
        "model": {"base_channels": M.BASE_CH, "jepa_channels": M.JEPA_CH, "jepa_projection": M.JEPA_PROJ,
                  "width": M.WIDTH, "dilations": list(M.DILATIONS), "r_max": M.R_MAX},
        "training": {"fit": list(T.FIT), "seeds": list(T.SEEDS), "split_seed": T.SPLIT_SEED, "val_places": T.N_VAL,
                     "epochs": T.EPOCHS, "batch": T.BATCH, "lr": T.LR, "weight_decay": T.WD,
                     "selection": "per run, the epoch with the lowest validation place-level AbsRel"},
        "primary": "P1 CONF_SYSTEM3: FUSE_BASE - FUSE_JEPA (three-seed ensembles), V1 paired place bootstrap + V2 "
                   "gate; P2 the same on CONF_VAL",
        "branches": ["JEPA_FUSION_SUPPORTED", "JEPA_FUSION_PARTIAL", "JEPA_FUSION_NOT_ESTABLISHED",
                     "JEPA_FUSION_HARMFUL"],
        "final_holdout_touched": False,
    }
    (root / "preregistration.json").write_text(json.dumps(prereg, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/PROTOCOL.locked.md").write_bytes((root / "PROTOCOL.md").read_bytes())
    (root / "audit/locked_at.txt").write_text(f"{prereg['locked_at']} {sha(root / 'preregistration.json')}\n")
    print("FN_LOCKED", prereg["locked_at"], sha(root / "preregistration.json"))


if __name__ == "__main__":
    main()
