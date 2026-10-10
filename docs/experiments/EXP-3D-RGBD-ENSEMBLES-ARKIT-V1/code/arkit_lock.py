"""EXP-3D-RGBD-ENSEMBLES-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a lock).
Run from the repository root:  python3 arkit_lock.py outputs/EXP-3D-RGBD-ENSEMBLES-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
DATA = Path("outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1").resolve()
RELEASED = Path.home() / "zonghan_o_mirror/_models/depthsplat/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"


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
    "experiment": "EXP-3D-RGBD-ENSEMBLES-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "evaluation_contract.json",
                                              "manifest_arkit_val80.json")},
    "data": {n: sha(DATA / n) for n in ("selection.json", "selection_extension.json", "candidate_lock.json",
                                        "candidate_lock_extension.json", "manifest_arkit_val.json",
                                        "manifest_arkit_val80.json", "preparation_integrity_val80.json")},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "geometry_seal.py", "score_analyze.py", "ds_predict.py",
                                                  "ds_finetune.py", "train_jepa.py")},
    "v1_checkpoints": {f"J{s}": sha(V1 / f"checkpoints/J{s}/depthsplat_jepa_final.pth") for s in (1, 2, 3)},
    "released_weights_sha256": sha(RELEASED),
    "primary": {"A1_COMPLETION_ENSEMBLE_VS_HARMONIC_FAR": "FAR: H3 - HJG",
                "A2_JEPA_ADDS_TO_ENSEMBLE_FAR": "FAR: HG - HJG",
                "B1_TTT_ENSEMBLE_VS_REPROJ7_ALL": "ALL: R7 - sqrt(R7 x TTT)",
                "B2_TTT_NEEDED_IN_ENSEMBLE_ALL": "ALL: sqrt(R7 x CACHE) - sqrt(R7 x TTT)"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by arkit_score.py score after every seal is complete",
              "ARKitScenes media: non-commercial research only, outside git, never redistributed; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
