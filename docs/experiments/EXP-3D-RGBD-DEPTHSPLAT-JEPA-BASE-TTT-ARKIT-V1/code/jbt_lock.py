"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite
a lock, and refuses once any JEPA-ARKIT result exists, so that no JA result can inform this design).
Run from the repository root:  python3 jbt_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
JEPA = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1").resolve()
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
if (JEPA / "scores").exists() or (JEPA / "results").exists() or (JEPA / "amendment_1/scene_table.json").exists():
    raise SystemExit("a JEPA-ARKIT result exists: too late to lock this design blind")
gpu_log = (JEPA / "audit/gpu.log").read_text()
if "TRAIN_END" not in gpu_log or "trained=3" not in gpu_log:
    raise SystemExit("JA training has not finished")
checkpoint = JEPA / "checkpoints/JA1/depthsplat_jepa_final.pth"
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "reused_seals.json")},
    "ja_checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint), "seed": 20261008,
                      "jepa_arkit_preregistration_sha256": sha(JEPA / "preregistration.json")},
    "released_weights_sha256": sha(RELEASED),
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "jepa_arkit_results_existed_at_lock": False,
    "settings": {"arms": "CACHE, TTT m = 1 (100 steps) from V1's ds_eval.py", "caps": {"JA": 3.0, "ZS3": 3.0, "ZS": 0.1},
                 "readout": "F7: ray depth where alpha >= 0.5 else REPROJ7_HARMONIC", "alpha_min": 0.5},
    "primary": {"Q1_JEPA_BASE_VS_RELEASED_TTT_ALL": "ALL: ZS_TTT - JA_TTT",
                "Q2_JEPA_TRAINING_HELPS_TTT_SAME_CAP_ALL": "ALL: ZS3_TTT - JA_TTT"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by jbt_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
