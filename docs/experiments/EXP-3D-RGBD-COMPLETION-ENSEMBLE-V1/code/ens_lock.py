"""EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a lock).
Run from the repository root:  python3 ens_lock.py outputs/EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
if (root / "preregistration.json").exists():
    raise SystemExit("already locked")
lock = {
    "experiment": "EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "evaluation_contract.json", "manifest_eval34.json")},
    "v1": {"preregistration_sha256": sha(V1 / "preregistration.json"), "ds_eval_sha256": sha(V1 / "code/ds_eval.py"),
           "score_analyze_sha256": sha(V1 / "code/score_analyze.py"),
           "checkpoints": {f"J{s}": sha(V1 / f"checkpoints/J{s}/depthsplat_jepa_final.pth") for s in (1, 2, 3)}},
    "primary": {"E1_ENSEMBLE_VS_HARMONIC_FAR": "FAR: H3 - HJG", "E2_JEPA_ADDS_TO_ENSEMBLE_FAR": "FAR: HG - HJG"},
    "branches": {"E1": "ENSEMBLE_CONSISTENT if SUPPORTED; ENSEMBLE_PARTIAL / ENSEMBLE_NOT_ESTABLISHED / ENSEMBLE_HARMFUL",
                 "E2": "JEPA_ADDS if SUPPORTED; JEPA_ADDS_PARTIAL / JEPA_NO_ADD / JEPA_HARMFUL"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by ens_score.py score after both seals are complete",
              "protected holdout never opened; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
