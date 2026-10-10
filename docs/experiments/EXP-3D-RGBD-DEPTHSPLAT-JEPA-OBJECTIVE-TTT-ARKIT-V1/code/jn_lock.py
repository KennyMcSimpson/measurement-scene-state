"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite
a lock or to lock once any JN / JK checkpoint or seal exists). Before locking it re-derives both training scripts from
OVERLAP-V1's train_jepa_arkit_overlap.py with the derivation scripts in code/ and requires byte-identical results, and
requires OVERLAP-V1's helper modules and hit table unchanged.
Run from the repository root:  python3 jn_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
PILOT = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1").resolve()
OVERLAP = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1").resolve()
JC = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1").resolve()
RELEASED = Path.home() / "zonghan_o_mirror/_models/depthsplat/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
code = root / "code"
if (root / "preregistration.json").exists():
    raise SystemExit("already locked")
if (root / "checkpoints").exists() or (root / "ds").exists() or (root / "scores").exists():
    raise SystemExit("JN / JK training or evaluation output exists")
with tempfile.TemporaryDirectory() as tmp:
    norm, cache = Path(tmp) / "norm.py", Path(tmp) / "cache.py"
    subprocess.run([sys.executable, str(code / "derive_train_norm.py"), str(OVERLAP / "code/train_jepa_arkit_overlap.py"),
                    str(norm)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(code / "derive_train_cache.py"), str(norm), str(cache)], check=True,
                   stdout=subprocess.DEVNULL)
    if sha(norm) != sha(code / "train_jepa_arkit_norm.py") or sha(cache) != sha(code / "train_jepa_arkit_cache.py"):
        raise SystemExit("the training scripts are not the derivations of OVERLAP-V1's script")
for name in ("ds_predict.py", "ds_finetune.py", "arkit_records.py"):
    if sha(code / name) != sha(OVERLAP / "code" / name):
        raise SystemExit(f"{name} must be OVERLAP-V1's")
if sha(root / "training/hit_shares_TRAIN.json") != sha(OVERLAP / "training/hit_shares_TRAIN.json"):
    raise SystemExit("the hit table must be OVERLAP-V1's")
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted(code.iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "reused_seals.json",
                                              "training/hit_shares_TRAIN.json", "training/streams_TRAIN.json")},
    "training_data": {n: sha(PILOT / n) for n in ("manifest_arkit_trainval.json", "candidate_lock.json")},
    "overlap_v1_preregistration_sha256": sha(OVERLAP / "preregistration.json"),
    "jc_preregistration_sha256": sha(JC / "preregistration.json"),
    "overlap_v1_training_script_sha256": sha(OVERLAP / "code/train_jepa_arkit_overlap.py"),
    "released_weights_sha256": sha(RELEASED),
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "design_basis": "motivated by the JBT and JC results (known at design time); locked before any JN / JK training "
                    "or evaluation",
    "settings": {"training": "accuracy-only objective (covered pixels, alpha >= 0.5, no gradient through the mask); "
                             "JN1: OVERLAP-V1 episodes (3 context frames, overlapping targets with hit share >= 0.05); "
                             "JK1: 3 context + 4 Core B stream frames -> primary queries 14 / 15; both seed 20261008, "
                             "4000 steps, lr 5e-6 / ViT 5e-7, warmup 200, cosine, AdamW wd 1e-4, clip 1.0, 2 targets, "
                             "scale cap 0.1, last step",
                 "evaluation": "CACHE + TTT m = 1 (100 steps), cap 0.1, V1's ds_eval.py, ARKIT_VAL (ENSEMBLES-ARKIT "
                               "episodes)", "alpha_min": 0.5},
    "primary": {"P1_ACCURACY_JEPA_VS_RELEASED_TTT_ALL": "ALL: ZS_TTT - JN_TTT",
                "P2_SEVEN_VIEW_ACCURACY_JEPA_VS_RELEASED_TTT_ALL": "ALL: ZS_TTT - JK_TTT"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by jn_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
