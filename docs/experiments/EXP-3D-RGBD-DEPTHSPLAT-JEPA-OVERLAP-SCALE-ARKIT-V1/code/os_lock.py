"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite
a lock or to lock once any OS checkpoint, seal or score exists). Before locking it re-derives the training script from
OVERLAP-V1's train_jepa_arkit_overlap.py with code/derive_train_overlap_scale.py and requires a byte-identical result,
and requires the record loader, helpers and merged hit table to be NS's / OVERLAP-V1's.
Run from the repository root:  python3 os_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OUT = Path("outputs").resolve()
V1 = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1"
OVERLAP = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1"
NS = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1"
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
if any((root / d).exists() for d in ("checkpoints", "ds", "scores")):
    raise SystemExit("OS training, evaluation or score output exists")
with tempfile.TemporaryDirectory() as tmp:
    derived = Path(tmp) / "scale.py"
    subprocess.run([sys.executable, str(code / "derive_train_overlap_scale.py"),
                    str(OVERLAP / "code/train_jepa_arkit_overlap.py"), str(derived)], check=True, stdout=subprocess.DEVNULL)
    if sha(derived) != sha(code / "train_jepa_arkit_overlap_scale.py"):
        raise SystemExit("train_jepa_arkit_overlap_scale.py is not the derivation of OVERLAP-V1's script")
if sha(code / "arkit_records_scale.py") != sha(NS / "code/arkit_records_scale.py"):
    raise SystemExit("arkit_records_scale.py must be NS's")
for name in ("ds_predict.py", "ds_finetune.py"):
    if sha(code / name) != sha(OVERLAP / "code" / name):
        raise SystemExit(f"{name} must be OVERLAP-V1's")
if sha(root / "training/hit_shares_SCALE.json") != sha(NS / "training/hit_shares_SCALE.json"):
    raise SystemExit("the hit table must be NS's merged table")
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted(code.iterdir()) if p.is_file()},
    "root_files": {"reused_seals.json": sha(root / "reused_seals.json"),
                   "eval_OVERLAP/frame_role_lock.json": sha(root / "eval_OVERLAP/frame_role_lock.json"),
                   "training/hit_shares_SCALE.json": sha(root / "training/hit_shares_SCALE.json")},
    "overlap_preregistration_sha256": sha(OVERLAP / "preregistration.json"),
    "overlap_results_sha256": sha(OVERLAP / "results/ARKIT160_OVERLAP_results.json"),
    "ns_preregistration_sha256": sha(NS / "preregistration.json"),
    "ns_results_sha256": sha(NS / "results/NS_results.json"),
    "released_weights_sha256": sha(RELEASED),
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "design_basis": "the NS result (known: more data did not change the accuracy-only feed-forward) and OVERLAP-V1's "
                    "JO completion (PARTIAL); one factor changed (training data 182 -> 502 scans); locked before any OS "
                    "training or evaluation",
    "settings": {"training": "JOL1-JOL3: train_jepa_arkit_overlap_scale.py (OVERLAP-V1 script, record source "
                             "arkit_records_scale), seeds 20261008-20261010, JO's settings (4000 steps, lr 5e-6 / ViT "
                             "5e-7, warmup 200, cosine, AdamW wd 1e-4, clip 1.0, 2 targets, min hit 0.05, scale cap 3.0, "
                             "last step)",
                 "evaluation": "V1's ds_eval.py, CTX arm, cap 3.0, ARKIT160_OVERLAP; only the ARKIT_VAL part scored",
                 "alpha_min": 0.5, "pooling": "at least two of three seeds cover; mean ray depth of the covering seeds"},
    "primary": {"P1_MORE_DATA_COMPLETION_FARH": "VAL80 FARH: JO - JOL",
                "P2_SCALED_COMPLETION_VS_HARMONIC_FARH": "VAL80 FARH: H3 - JOL"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by os_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
