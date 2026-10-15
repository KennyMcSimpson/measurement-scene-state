"""EXP-3D-READOUT-METHOD-V1: the preregistration lock (before any evaluation fold is scored).
Hashes the protocol, the code, the input caches and the frozen KF table; writes PREREGISTRATION.json, a copy of the
locked protocol and audit/locked_at.txt. Refuses when results/phase12 exists.
Run from the repository root:  python rm_lock.py ROOT
"""

import hashlib
import json
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    if (root / "results/phase12").exists() or (root / "PREREGISTRATION.json").exists():
        raise SystemExit("cannot lock: results or a preregistration exist already")
    prereg = {
        "experiment": "EXP-3D-READOUT-METHOD-V1", "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "protocol_sha256": sha(root / "PROTOCOL.md"),
        "code": {p.name: sha(p) for p in sorted((root / "code").glob("*.py"))},
        "caches": {p.name: sha(p) for p in sorted((root / "cache").glob("*"))},
        "kf_table_sha256": "15100550ca6b39763244a69405ec88f8eede4faaa700aff4619557ee1c7e533c",
        "folds": ["CONF_SYSTEM3", "CONF_VAL", "CAJU", "ARKIT_SYSTEM4"],
        "training_only": ["FIT_SYSTEM", "FIT_SYSTEM2"],
        "phase1_methods": ["B0", "B1", "B2A", "B2B", "B3", "B4", "B5", "B6", "B6W", "B7"],
        "phase2_methods": ["SYS_A", "SYS_B", "SYS_C", "SYS_C_ON_B", "SYS_B_ON_C", "R7", "CORE_A", "B0"],
        "gate_a": {"comparison": "B7 vs S* = the lowest pooled place mean among B2A, B2B, B3, B4, B5",
                   "ci_lower_gt": 0.0, "min_mean_gain": 0.005, "max_top1_share": 0.5, "median_gain_gt": 0.0,
                   "uncovered_pixel_absrel_lower_in_every_fold": True,
                   "label": ["V2_METHOD_NECESSITY=SUPPORTED", "V2_METHOD_NECESSITY=NOT_ESTABLISHED"]},
        "gate_b": {"comparison": "SYS_C vs SYS_B (matched readouts)", "ci_lower_gt": 0.0, "min_mean_gain": 0.002,
                   "median_gain_gt": 0.0, "max_top1_share": 0.5,
                   "label": ["TTT_ADDED_VALUE=SUPPORTED", "TTT_ADDED_VALUE=NOT_ESTABLISHED"]},
        "bootstrap": {"draws": 10000, "seed": 20260928, "unit": "place", "source": "score_analyze.contrast (V1)"},
        "smoke_disclosure": "one code smoke run before the lock: train FIT_SYSTEM, score FIT_SYSTEM2 (a training "
                            "cohort), 1 seed, 1 epoch; no evaluation fold was scored",
        "fresh_data_used": False, "final_holdout_touched": False,
    }
    (root / "PREREGISTRATION.json").write_text(json.dumps(prereg, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/PROTOCOL.locked.md").write_bytes((root / "PROTOCOL.md").read_bytes())
    (root / "audit/locked_at.txt").write_text(f"{prereg['locked_at']} {sha(root / 'PREREGISTRATION.json')}\n")
    print("RM_LOCKED", prereg["locked_at"], sha(root / "PREREGISTRATION.json"))


if __name__ == "__main__":
    main()
