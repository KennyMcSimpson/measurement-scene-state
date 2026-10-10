"""EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a
lock or to lock after any prediction exists). Run from the repository root:
  python3 far_lock.py outputs/EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1").resolve()
OVERLAP = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1").resolve()
JEPA = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1").resolve()
SELECTOR = Path("outputs/EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1").resolve()
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
if (root / "raw").exists() or (root / "ds").exists() or (root / "scores").exists():
    raise SystemExit("a prediction or score exists")
lock = {
    "experiment": "EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "evaluation_contract.json", "selector_seal.json")},
    "selector_experiment_preregistration_sha256": sha(SELECTOR / "preregistration.json"),
    "models": {"GA_contract_sha256": sha(CARRIER / "evaluation_contract.json"),
               "JO_checkpoints": {f"JO{s}": sha(OVERLAP / f"checkpoints/JO{s}/depthsplat_jepa_final.pth") for s in (1, 2, 3)},
               "JA_checkpoints": {f"JA{s}": sha(JEPA / f"checkpoints/JA{s}/depthsplat_jepa_final.pth") for s in (1, 2, 3)}},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "far_view_results_existed": False,
    "settings": {"selector": "the SELECTOR-V1 rule and seal: learned where S >= 1.2", "queries": "primary_query (14, 15)",
                 "opacity_min": 0.5, "alpha_min": 0.5, "region": "FARH"},
    "primary": {"P1_SELECTED_CARRIER_VS_HARMONIC_FAR_VIEWS": "FARH: H3 - SEL_GAN"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by far_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit/locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                          f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
