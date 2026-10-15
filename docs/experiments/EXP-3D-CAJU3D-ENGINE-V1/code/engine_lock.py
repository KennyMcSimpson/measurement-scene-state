"""EXP-3D-CAJU3D-ENGINE-V1: the preregistration lock (before any engine run of this experiment and before any
ground truth is read). Writes preregistration.json (hashes of the protocol, the code, the root files, the imported V1 /
CAJU Gate 0 files and the frozen project sources; the constants) and audit/locked_at.txt, and keeps a copy of the
locked protocol. Refuses when an engine run, a score, a result or a selection exists.
Run through cv_host_run.py from the repository root (CUDA hidden):  python engine_lock.py ROOT
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_logic as L  # noqa: E402

OUT = Path("outputs").resolve()
IMPORTED = ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_eval.py", "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_predict.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_finetune.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/train_jepa.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/score_analyze.py", "EXP-3D-CAJU-HEADROOM-V1/code/caju_logic.py")
FROZEN = ("src/mcss/measurement_planning/information.py", "src/mcss/mechanism_pilot/direct_capacity_statistics.py",
          "src/mcss/mechanism_pilot/geometry_carrier_statistics.py")
ROOT_FILES = ("PROTOCOL.md", "frame_role_lock.json", "reused_seals.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(root):
    for name in ("engine", "scores", "results", "selection.json", "preregistration.json"):
        if (root / name).exists():
            raise SystemExit(f"cannot lock: {name} exists already")


def main():
    root = Path(sys.argv[1]).resolve()
    guard(root)
    prereg = {
        "experiment": "EXP-3D-CAJU3D-ENGINE-V1", "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root_files": {n: sha(root / n) for n in ROOT_FILES},
        "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.suffix in (".py", ".sh")},
        "imported": {rel: sha(OUT / rel) for rel in IMPORTED},
        "frozen_sources": {rel: sha(rel) for rel in FROZEN},
        "checkpoints": {k: sha(v) for k, v in L.CHECKPOINTS.items()},
        "constants": {k: getattr(L, k) for k in ("SCALE_MAX", "STEPS", "MULT", "SEED", "ALPHA_MIN", "VARIANTS",
                                                 "DEV_ARMS", "CANDIDATES", "COMPARATOR", "INC", "DEV_SEED",
                                                 "DEV_PER_COHORT")},
        "selection": "lowest DEV mean place AbsRel (Core B readout) among CANDIDATES; STOP unless it beats ZS_RAW",
        "primary": "CONF_SYSTEM3: CORE_B - C* (Core B readout), V1 paired place bootstrap + V2 gate; "
                   "P2 = the same on CONF_VAL (replication)",
        "branches": ["CAJU3D_ENGINE_SUPPORTED", "CAJU3D_ENGINE_PARTIAL", "CAJU3D_ENGINE_NOT_ESTABLISHED",
                     "CAJU3D_ENGINE_HARMFUL", "STOP_AT_DEV"],
        "final_holdout_touched": False,
    }
    (root / "preregistration.json").write_text(json.dumps(prereg, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/PROTOCOL.locked.md").write_bytes((root / "PROTOCOL.md").read_bytes())
    (root / "audit/locked_at.txt").write_text(f"{prereg['locked_at']} {sha(root / 'preregistration.json')}\n")
    print("ENGINE_LOCKED", prereg["locked_at"], sha(root / "preregistration.json"))


if __name__ == "__main__":
    main()
