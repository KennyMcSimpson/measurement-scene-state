"""EXP-3D-RGBD-TTT-ENSEMBLE-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a lock).
Run from the repository root:  python3 tttens_lock.py outputs/EXP-3D-RGBD-TTT-ENSEMBLE-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
ARMS = Path("outputs/EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1/raw/eval/EVAL34/prediction_seal.json").resolve()
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
    "experiment": "EXP-3D-RGBD-TTT-ENSEMBLE-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "manifest_eval34.json")},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "geometry_seal.py", "score_analyze.py", "ds_predict.py",
                                                  "ds_finetune.py", "train_jepa.py")},
    "arms_seal": {"path": str(ARMS), "sha256": sha(ARMS)},
    "released_weights_sha256": sha(RELEASED),
    "settings": {"builder": "ZS (released, scale cap 0.1)", "arms": ["CACHE", "TTT"], "ttt_mult": 1.0, "ttt_steps": 100,
                 "alpha_min": 0.5},
    "primary": {"T1_TTT_ENSEMBLE_VS_REPROJ7_ALL": "ALL: R7 - sqrt(R7 x TTT)",
                "T2_TTT_NEEDED_IN_ENSEMBLE_ALL": "ALL: sqrt(R7 x CACHE) - sqrt(R7 x TTT)"},
    "branches": {"T1": "TTT_ENSEMBLE_CONSISTENT if SUPPORTED; TTT_ENSEMBLE_PARTIAL / _NOT_ESTABLISHED / _HARMFUL",
                 "T2": "TTT_NEEDED if SUPPORTED; TTT_NEEDED_PARTIAL / TTT_NOT_NEEDED / TTT_HARMFUL"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by tttens_score.py score after both seals are complete",
              "protected holdout never opened; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
