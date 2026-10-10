"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite
a lock, to lock once any NF checkpoint, seal or score exists, or once V3 has scored). Requires the training script and its
helpers to be byte-identical to OBJECTIVE-TTT's (the ones that trained JN1) and the hit table to be the same.
Run from the repository root:  python3 nf_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
PILOT = Path("outputs/EXP-3D-RGBD-ARKIT-DATA-V1").resolve()
OBJ = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1").resolve()
V3 = Path("outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3").resolve()
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
    raise SystemExit("NF training, evaluation or score output exists")
if (V3 / "scores").exists():
    raise SystemExit("V3 has scored: ARKIT_SYSTEM3 query depth was read")
for name in ("train_jepa_arkit_norm.py", "ds_predict.py", "ds_finetune.py", "arkit_records.py"):
    if sha(code / name) != sha(OBJ / "code" / name):
        raise SystemExit(f"{name} must be OBJECTIVE-TTT's")
if sha(root / "training/hit_shares_TRAIN.json") != sha(OBJ / "training/hit_shares_TRAIN.json"):
    raise SystemExit("the hit table must be OBJECTIVE-TTT's")
reused = json.loads((root / "reused_seals.json").read_text())
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted(code.iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "reused_seals.json",
                                              "training/hit_shares_TRAIN.json")},
    "training_data": {n: sha(PILOT / n) for n in ("manifest_arkit_trainval.json", "candidate_lock.json")},
    "objective_preregistration_sha256": sha(OBJ / "preregistration.json"),
    "objective_results_sha256": sha(OBJ / "results/ARKIT_VAL_results.json"),
    "v3_preregistration_sha256": sha(V3 / "preregistration.json"),
    "v3_scored_at_lock": False,
    "released_weights_sha256": sha(RELEASED),
    "reused_checkpoints": {k: v["sha256"] for k, v in reused["checkpoints"].items()},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "design_basis": "S3 of EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1 (known at design time); locked before any "
                    "NF training or evaluation and before V3 read any ARKIT_SYSTEM3 query depth",
    "settings": {"training": "JN2 / JN3: train_jepa_arkit_norm.py with JN1's settings (4000 steps, lr 5e-6 / ViT 5e-7, "
                             "warmup 200, cosine, AdamW wd 1e-4, clip 1.0, 2 targets, scale cap 0.1, last step), "
                             "seeds 20261009 / 20261010",
                 "evaluation": "V1's ds_eval.py, CACHE arm only (3 context + 4 stream views), cap 0.1, ARKIT_SYSTEM3 "
                               "(V3 episodes): ZS (released), JN1, JN2, JN3, JK1", "alpha_min": 0.5,
                 "recipe_readout": "JNS = per-query mean over JN1-JN3 of the AbsRel"},
    "primary": {"P1_JN1_FEEDFORWARD_VS_RELEASED_ALL": "ALL: ZS_CACHE - JN1_CACHE",
                "P2_RECIPE_FEEDFORWARD_VS_RELEASED_ALL": "ALL: ZS_CACHE - JNS_CACHE"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by nf_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
