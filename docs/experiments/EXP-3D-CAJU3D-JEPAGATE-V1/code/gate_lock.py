"""EXP-3D-CAJU3D-JEPAGATE-V1: the preregistration lock (after the reverse JEPA is trained, before any gate run of
this experiment and before any ground truth is read). Hashes the protocol, the code, the root files (copied from
ENGINE-V1), the trained model and its log, the imported files; keeps a copy of the locked protocol. Refuses when a gate
run, a score, a result or a decision exists.
Run through cv_host_run.py from the repository root (CUDA hidden):  python gate_lock.py ROOT
"""

import hashlib
import json
import sys
import time
from pathlib import Path

OUT = Path("outputs").resolve()
IMPORTED = ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_eval.py", "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_predict.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/ds_finetune.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/train_jepa.py",
            "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/score_analyze.py", "EXP-3D-CAJU-HEADROOM-V1/code/caju_logic.py",
            "EXP-3D-CAJU3D-ENGINE-V1/code/engine_core.py", "EXP-3D-CAJU3D-ENGINE-V1/code/engine_score.py",
            "EXP-3D-CAJU3D-ENGINE-V1/code/engine_logic.py")
ROOT_FILES = ("PROTOCOL.md", "frame_role_lock.json", "reused_seals.json", "model/hjepa_final.pth",
              "model/train_log.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(root):
    for name in ("gate", "scores", "results", "decision.json", "preregistration.json"):
        if (root / name).exists():
            raise SystemExit(f"cannot lock: {name} exists already")


def main():
    root = Path(sys.argv[1]).resolve()
    guard(root)
    train_log = json.loads((root / "model/train_log.json").read_text())
    prereg = {
        "experiment": "EXP-3D-CAJU3D-JEPAGATE-V1", "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root_files": {n: sha(root / n) for n in ROOT_FILES},
        "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.suffix in (".py", ".sh")},
        "imported": {rel: sha(OUT / rel) for rel in IMPORTED},
        "jepa_training": {"steps": train_log["args"]["steps"], "final_eval": train_log["eval"][-1],
                          "train_scans": train_log["train_scans"], "dev_monitor_scans": train_log["dev_scans"]},
        "constants": {"share": 0.10, "ttt_steps": 100, "scale_max": 0.1,
                      "inc": {"cos_lo": 0.1, "cos_hi": 0.4, "edge_max": 0.1},
                      "dev_arms": ["JEPA", "DEPTH", "INC", "RANDOM"], "conf_arms": ["JEPA", "DEPTH"]},
        "dev_rule": "GO iff GATE_JEPA's DEV mean (Core B readout) is below both CORE_B and GATE_DEPTH",
        "primary": "CONF_SYSTEM3: GATE_DEPTH - GATE_JEPA, V1 paired place bootstrap + V2 gate; P2 the same on CONF_VAL",
        "branches": ["JEPA_ADDED_VALUE_SUPPORTED", "JEPA_ADDED_VALUE_PARTIAL", "JEPA_ADDED_VALUE_NOT_ESTABLISHED",
                     "JEPA_ADDED_VALUE_HARMFUL", "STOP_AT_DEV"],
        "final_holdout_touched": False,
    }
    (root / "preregistration.json").write_text(json.dumps(prereg, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/PROTOCOL.locked.md").write_bytes((root / "PROTOCOL.md").read_bytes())
    (root / "audit/locked_at.txt").write_text(f"{prereg['locked_at']} {sha(root / 'preregistration.json')}\n")
    print("GATE_LOCKED", prereg["locked_at"], sha(root / "preregistration.json"))


if __name__ == "__main__":
    main()
