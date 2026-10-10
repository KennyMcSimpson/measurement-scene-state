"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a
lock; records whether any JEPA-ARKIT-V1 result existed at lock time).
Run from the repository root:  python3 overlap_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
JEPA = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1").resolve()
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
seen = [str(p.relative_to(JEPA)) for p in (JEPA / "scores", JEPA / "results", JEPA / "amendment_1/scene_table.json")
        if p.exists()]
table = json.loads((root / "training/hit_shares_TRAIN.json").read_text())
eligible = [share >= 0.05 for roles in table.values() for frames in roles.values() for share in frames.values()]
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "evaluation_contract.json",
                                              "manifest_arkit160_overlap.json", "training/hit_shares_TRAIN.json")},
    "training_data": {n: sha(PILOT / n) for n in ("manifest_arkit_trainval.json", "candidate_lock.json")},
    "hit_table": {"pairs": len(eligible), "eligible_share": sum(eligible) / len(eligible), "min_hit": 0.05},
    "jepa_arkit_v1": {"preregistration_sha256": sha(JEPA / "preregistration.json"),
                      "train_script_sha256": sha(JEPA / "code/train_jepa_arkit.py"),
                      "checkpoints": {f"JA{s}": sha(JEPA / f"checkpoints/JA{s}/depthsplat_jepa_final.pth")
                                      for s in (1, 2, 3)},
                      "results_present_at_lock": seen},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "v1_checkpoints": {f"J{s}": sha(V1 / f"checkpoints/J{s}/depthsplat_jepa_final.pth") for s in (1, 2, 3)},
    "released_weights_sha256": sha(RELEASED),
    "settings": {"training": "JEPA-ARKIT-V1 recipe (seeds 20261008-10, 4000 steps, lr 5e-6 / ViT 5e-7, warmup 200, "
                             "cosine, AdamW wd 1e-4, clip 1.0, 2 targets, relative L1 raw depth + 0.5 RGB L1, cap 3.0) "
                             "with targets restricted to non-context frames of context hit share >= 0.05",
                 "queries": "frames 6 and 7 (next free of 9, 11, 12 where a record uses one)", "alpha_min": 0.5, "g1_opacity_min": 0.5, "zero_hit_constant_m": 1.05},
    "primary": {"O1_OVERLAP_JEPA_VS_HARMONIC_FARH": "FARH: H3 - JO",
                "O2_OVERLAP_TRAINING_VS_ORIGINAL_RECIPE_FARH": "FARH: JA - JO"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by overlap_score.py score after every seal is complete",
              "a failed training seed stops the run (reported); ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"), "JEPA-ARKIT results present:", seen)
