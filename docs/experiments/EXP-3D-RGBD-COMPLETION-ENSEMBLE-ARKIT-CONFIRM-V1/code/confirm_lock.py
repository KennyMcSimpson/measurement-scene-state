"""EXP-3D-RGBD-COMPLETION-ENSEMBLE-ARKIT-CONFIRM-V1: preregistration.json and audit/locked_at.txt.
Run from the repository root:  python3 confirm_lock.py outputs/EXP-3D-RGBD-COMPLETION-ENSEMBLE-ARKIT-CONFIRM-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

DATA = Path("outputs/EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1").resolve()
V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
if (root / "preregistration.json").exists():
    raise SystemExit("already locked")
lock = {
    "experiment": "EXP-3D-RGBD-COMPLETION-ENSEMBLE-ARKIT-CONFIRM-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "evaluation_contract.json",
                                              "manifest_arkit_confirm.json")},
    "data": {n: sha(DATA / n) for n in ("selection.json", "downloads.json", "candidate_lock.json",
                                        "manifest_arkit_confirm.json", "preparation_integrity.json")},
    "v1_score_analyze_sha256": sha(V1 / "code/score_analyze.py"),
    "primary": {"C1_NORMALIZED_ENSEMBLE_VS_HARMONIC_FAR": "FAR: H3 - (H3 + G1N) / 2"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by confirm_exp.py score after the arms seal is complete",
              "ARKitScenes media: non-commercial research only, outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
