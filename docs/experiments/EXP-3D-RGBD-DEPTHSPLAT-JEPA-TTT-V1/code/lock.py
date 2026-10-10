"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1: write preregistration.json (sha256 of every input the formal run depends on)
and audit/locked_at.txt. Refuses to overwrite an existing lock. Run from the repository root:
  python3 lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
DS_REPO = HOME / "cv_work/depthsplat/depthsplat"
DINOV2 = HOME / ".cache/torch/hub/facebookresearch_dinov2_main"
RELEASED = HOME / "zonghan_o_mirror/_models/depthsplat/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"
INPUTS = [
    "scripts/evaluate_rgbd_carrier_arms.py",
    "scripts/evaluate_rgbd_long_train_eval.py",
    "scripts/run_rgbd_geometric_baselines.py",
    "scripts/run_rgbd_inpaint_baselines.py",
    "src/mcss/measurement_planning/geometry.py",
    "src/mcss/measurement_planning/information.py",
    "src/mcss/mechanism_pilot/rgbd_stream_write.py",
    "src/mcss/mechanism_pilot/direct_capacity_statistics.py",
    "src/mcss/mechanism_pilot/geometry_carrier_statistics.py",
    "outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7/scene_split.json",
    "outputs/EXP-3D-RGBD-SEGMENT-DATA-V1/manifest_seg.json",
    "outputs/EXP-3D-RGBD-EVAL-V5-DATA/manifest_eval_v5.json",
    "outputs/EXP-3D-RGBD-TRAIN-EXT-DATA/manifest_train_ext.json",
    "outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7/stream_roles.json",
    "outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6/stream_roles.json",
]


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha(directory):
    files = sorted(p for p in Path(directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return hashlib.sha256("".join(f"{p.relative_to(directory)} {sha(p)}\n" for p in files).encode()).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    if (root / "preregistration.json").exists():
        raise SystemExit("already locked")
    ds_head = subprocess.run(["git", "-C", str(DS_REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    ds_dirty = subprocess.run(["git", "-C", str(DS_REPO), "status", "--porcelain", "--untracked-files=no"],
                              capture_output=True, text=True).stdout
    lock = {
        "experiment": "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1",
        "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "protocol_sha256": sha(root / "PROTOCOL.md"),
        "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
        "root_files": {name: sha(root / name) for name in ("frame_role_lock.json", "evaluation_contract.json",
                                                            "manifest_fresh78.json", "ttt_selection.json")},
        "repository_inputs": {rel: sha(rel) for rel in INPUTS},
        "dev_manifest": {"path": str(HOME / "cv_work/depthsplat/probe_dev/manifest_dev.json"),
                         "sha256": sha(HOME / "cv_work/depthsplat/probe_dev/manifest_dev.json")},
        "dev_arms_seal": {"path": str(HOME / "cv_work/depthsplat/probe_dev/raw/eval/DEV/prediction_seal.json"),
                          "sha256": sha(HOME / "cv_work/depthsplat/probe_dev/raw/eval/DEV/prediction_seal.json")},
        "depthsplat": {"repository": str(DS_REPO), "commit": ds_head.strip(), "tracked_changes": ds_dirty.strip(),
                       "released_weights": str(RELEASED), "released_weights_sha256": sha(RELEASED),
                       "dinov2_code_tree_sha256": tree_sha(DINOV2)},
        "settings": {
            "training": {"seeds": [20261008, 20261009, 20261010], "steps": 4000, "lr": 5e-6, "vit_lr": 5e-7,
                         "warmup": 200, "schedule": "cosine to 0", "weight_decay": 1e-4, "grad_clip": 1.0,
                         "targets_per_step": 2, "rgb_weight": 0.5, "scale_max": 3.0, "input": [256, 320],
                         "render": [128, 160], "data": "TRAIN72 + SEG-V1 (312 records)", "final_step_used": True},
            "ttt": {"steps": 100, "builder": "ZS", "multiplier_grid_dev": [1, 3, 10],
                    "selected_multiplier": json.loads((root / "ttt_selection.json").read_text())["selected_mult"]},
            "alpha_min": 0.5, "near_px": 8, "bootstrap": {"draws": 10000, "seed": 20260928, "unit": "scene"},
        },
        "primary": {
            "A1_JEPA_VS_HARMONIC_FAR": "FAR AbsRel: REPROJ_HARMONIC - J_CTX_F (J = mean of J1..J3 per scene)",
            "B1_TTT_BEYOND_CACHE_ALL": "ALL AbsRel: ZS_CACHE_F7 - ZS_TTT_F7",
            "B2_TTT_VS_REPROJ7_ALL": "ALL AbsRel: REPROJ7_HARMONIC - ZS_TTT_F7 (the TTT state where alpha >= 0.5, "
                                     "REPROJ7_HARMONIC elsewhere)",
        },
        "branches": {
            "core_a": "A_SUPPORTED if A1 passes the V2 gate; A_PARTIAL if mean > 0 and CI low > 0 only; A_HARMFUL "
                      "if CI high < 0; else A_NOT_ESTABLISHED",
            "core_b": "B_SUPPORTED_PRACTICAL if B1 and B2 pass; B_MECHANISM_ONLY if only B1 passes; B_PARTIAL / "
                      "B_HARMFUL / B_NOT_ESTABLISHED by B1's status otherwise",
        },
        "rules": ["no setting changes after this lock; amendments only as appended, dated records",
                  "query depth is read only by score_analyze.py score, after every seal is complete",
                  "a failed training seed is reported; the analysis then uses the completed seeds if at least two "
                  "completed, without any setting change", "protected holdout never opened; nothing pushed"],
    }
    (root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit" / "locked_at.txt").write_text(lock["locked_at"] + " sha256(preregistration.json)=" +
                                                  sha(root / "preregistration.json") + "\n")
    print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))


if __name__ == "__main__":
    main()
