"""EXP-3D-CAJU3D-KFUSE-V1: the preregistration lock (before any H-JEPA query feature of this experiment is computed,
before the tables are fitted and before any confirmation truth is read). Hashes the protocol, the code, the root
files, the frozen H-JEPA and the imported files; keeps a copy of the locked protocol. Refuses when features, tables,
scores or results exist.
Run through cv_host_run.py from the repository root (CUDA hidden):  python kf_lock.py ROOT
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kf_logic as K  # noqa: E402

OUT = Path("outputs").resolve()
IMPORTED = ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_eval.py", "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_predict.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_finetune.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/score_analyze.py", "EXP-3D-CAJU-HEADROOM-V1/code/caju_logic.py",
            "EXP-3D-CAJU3D-ENGINE-V1/code/engine_score.py", "EXP-3D-CAJU3D-ENGINE-V1/code/engine_logic.py",
            "EXP-3D-CAJU3D-JEPAGATE-V1/code/hjepa.py", "EXP-3D-CAJU3D-JEPAGATE-V1/model/hjepa_final.pth",
            "EXP-3D-CAJU3D-JEPAGATE-V1/preregistration.json")
ROOT_FILES = ("PROTOCOL.md", "frame_role_lock.json", "reused_seals.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    for name in ("jepa", "fusion_tables.json", "scores", "results", "preregistration.json"):
        if (root / name).exists():
            raise SystemExit(f"cannot lock: {name} exists already")
    prereg = {
        "experiment": "EXP-3D-CAJU3D-KFUSE-V1", "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root_files": {n: sha(root / n) for n in ROOT_FILES},
        "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.suffix in (".py", ".sh")},
        "imported": {rel: sha(OUT / rel) for rel in IMPORTED},
        "constants": {"alpha_min": K.ALPHA_MIN, "weight_grid": K.WEIGHTS.tolist(), "feature_bins": K.FEATURE_BINS,
                      "arms": {a: list(v) for a, v in K.ARMS.items()}},
        "fit": "FIT_SYSTEM + FIT_SYSTEM2: quantile edges on the covered pixels; per bin the grid weight minimizing the "
               "place-weighted AbsRel; frozen and hashed before any confirmation truth is read",
        "primary": "P1 CONF_SYSTEM3: KF_BASE - KF_JEPA, V1 paired place bootstrap + V2 gate; P2 the same on CONF_VAL",
        "branches": ["JEPA_KALMAN_SUPPORTED", "JEPA_KALMAN_PARTIAL", "JEPA_KALMAN_NOT_ESTABLISHED",
                     "JEPA_KALMAN_HARMFUL"],
        "final_holdout_touched": False,
    }
    (root / "preregistration.json").write_text(json.dumps(prereg, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/PROTOCOL.locked.md").write_bytes((root / "PROTOCOL.md").read_bytes())
    (root / "audit/locked_at.txt").write_text(f"{prereg['locked_at']} {sha(root / 'preregistration.json')}\n")
    print("KF_LOCKED", prereg["locked_at"], sha(root / "preregistration.json"))


if __name__ == "__main__":
    main()
