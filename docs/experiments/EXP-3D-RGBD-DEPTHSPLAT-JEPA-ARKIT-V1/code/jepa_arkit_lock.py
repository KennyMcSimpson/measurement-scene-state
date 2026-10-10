"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a lock).
Run from the repository root:  python3 jepa_arkit_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
PILOT = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1").resolve()
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
train = json.loads((PILOT / "manifest_arkit_trainval.json").read_text())["scenes"]
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "evaluation_contract.json", "manifest_arkit160.json")},
    "training_data": {n: sha(PILOT / n) for n in ("selection.json", "downloads.json", "candidate_lock.json",
                                                  "manifest_arkit_trainval.json", "preparation_integrity.json")},
    "training_counts": {split: sum(r["split"] == split for r in train) for split in ("DEV", "TRAIN")},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "train_jepa.py", "score_analyze.py")},
    "v1_checkpoints": {f"J{s}": sha(V1 / f"checkpoints/J{s}/depthsplat_jepa_final.pth") for s in (1, 2, 3)},
    "released_weights_sha256": sha(RELEASED),
    "settings": {"training": "V1 recipe: seeds 20261008-10, 4000 steps, lr 5e-6 (ViT 5e-7), warmup 200, cosine, "
                             "AdamW wd 1e-4, clip 1.0, 2 targets, relative L1 raw depth + 0.5 RGB L1, scale cap 3.0",
                 "alpha_min": 0.5, "g1_opacity_min": 0.5},
    "primary": {"R1_REAL_JEPA_VS_HARMONIC_FAR": "FAR: H3 - JA", "R2_REAL_VS_SYNTHETIC_TRAINING_FAR": "FAR: J - JA"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by jepa_arkit_score.py score after every seal is complete",
              "a failed training seed stops the run (reported); ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"), lock["training_counts"])
