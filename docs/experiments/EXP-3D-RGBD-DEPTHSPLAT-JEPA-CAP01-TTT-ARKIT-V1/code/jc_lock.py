"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a
lock or to lock once any JC checkpoint or seal exists).
Run from the repository root:  python3 jc_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
PILOT = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1").resolve()
OVERLAP = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1").resolve()
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
if (root / "checkpoints").exists() or (root / "ds").exists() or (root / "scores").exists():
    raise SystemExit("JC training or evaluation output exists")
if sha(root / "code/train_jepa_arkit_overlap.py") != sha(OVERLAP / "code/train_jepa_arkit_overlap.py"):
    raise SystemExit("the training script must be OVERLAP-V1's")
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "reused_seals.json", "training/hit_shares_TRAIN.json")},
    "training_data": {n: sha(PILOT / n) for n in ("manifest_arkit_trainval.json", "candidate_lock.json")},
    "overlap_v1_preregistration_sha256": sha(OVERLAP / "preregistration.json"),
    "released_weights_sha256": sha(RELEASED),
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "design_basis": "motivated by the JBT and OVERLAP-V1 results (known at design time); locked before any JC training "
                    "or evaluation",
    "settings": {"training": "OVERLAP-V1 recipe (train_jepa_arkit_overlap.py, min hit 0.05) with scale cap 0.1, seed "
                             "20261008 only (JC1), 4000 steps, lr 5e-6 / ViT 5e-7, warmup 200, cosine, AdamW wd 1e-4, "
                             "clip 1.0, 2 targets",
                 "evaluation": "CACHE + TTT m = 1 (100 steps), cap 0.1, V1's ds_eval.py, ARKIT_VAL (ENSEMBLES-ARKIT "
                               "episodes)", "alpha_min": 0.5},
    "primary": {"P1_JEPA_CAP01_VS_RELEASED_TTT_ALL": "ALL: ZS_TTT - JC_TTT",
                "P2_JEPA_CAP01_VS_RELEASED_CACHE_ALL": "ALL: ZS_CACHE - JC_CACHE"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by jc_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
