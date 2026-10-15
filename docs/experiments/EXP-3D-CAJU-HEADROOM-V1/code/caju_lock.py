"""EXP-3D-CAJU-HEADROOM-V1: the preregistration lock (before any replay and before any ground truth is read).

Writes preregistration.json with the sha256 of the protocol, every code file of this experiment, the V1 files the
replay imports, the frozen geometry sources, the root files and the geometry seals, plus the frozen constants of
caju_logic; then audit/locked_at.txt. Refuses when a replay, equivalence, score or result output exists, or when a
geometry seal is missing.
Run through cv_host_run.py from the repository root (CUDA hidden):  python caju_lock.py ROOT
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caju_logic as L  # noqa: E402

V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code").resolve()
FROZEN = ("scripts/run_rgbd_geometric_baselines.py", "scripts/run_rgbd_inpaint_baselines.py",
          "src/mcss/measurement_planning/geometry.py", "src/mcss/measurement_planning/information.py",
          "src/mcss/mechanism_pilot/rgbd_stream_write.py",
          "src/mcss/mechanism_pilot/direct_capacity_statistics.py")
ROOT_FILES = ("PROTOCOL.md", "frame_role_lock.json", "scene_manifest.json", "episode_manifest.json",
              "action_contract.json", "information_contract.json", "code_audit.md", "jepa_feature_contract.md")
COHORTS = ("CAJU_PILOT", "CAJU_SEL", "CAJU_EVAL", "CAJU_COMPAT")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(root):
    """The lock must come before every replay, equivalence run, score and result."""
    for name in ("replay", "equivalence", "scores", "results", "pilot", "fixed_action_selection.json",
                 "preregistration.json"):
        if (root / name).exists():
            raise SystemExit(f"cannot lock: {name} exists already")
    for cohort in COHORTS:
        if not (root / "geometry" / f"geometry_seal_{cohort}.json").is_file():
            raise SystemExit(f"cannot lock: geometry of {cohort} is not sealed")


def main():
    root = Path(sys.argv[1]).resolve()
    guard(root)
    constants = {k: getattr(L, k) for k in ("ACTIONS", "STEPS", "ALPHA_MIN", "SELECTION_SEED", "N_SEL", "N_EVAL",
                                            "DRAWS", "BOOT_SEED", "DELTA_MIN", "EPISODE_BAND", "TOP1_MAX",
                                            "COMPUTE_SAVING_MIN", "COMPUTE_LOSS_MEAN_MAX", "COMPUTE_LOSS_CI_MAX",
                                            "FIXED_RULE_TOLERANCE", "RELIABLE_ERROR")}
    prereg = {
        "experiment": "EXP-3D-CAJU-HEADROOM-V1", "gate": "CAJU-V1 Gate 0",
        "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root_files": {name: sha(root / name) for name in ROOT_FILES},
        "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.suffix in (".py", ".sh")},
        "v1_imported": {name: sha(V1 / name) for name in ("ds_eval.py", "ds_predict.py", "ds_finetune.py",
                                                         "train_jepa.py")},
        "frozen_sources": {rel: sha(rel) for rel in FROZEN},
        "geometry_seals": {c: sha(root / "geometry" / f"geometry_seal_{c}.json") for c in COHORTS},
        "constants": constants,
        "primary": "PRIMARY_XORACLE_MINUS_FIX on CAJU_EVAL places: mean >= DELTA_MIN, 95% place-bootstrap CI low > 0, "
                   "every leave-one-place-out mean > 0, positive top-1 place share <= 0.5",
        "branches": ["QUALITY_HEADROOM_SUPPORTED", "COMPUTE_HEADROOM_SUPPORTED",
                     "NOT_ESTABLISHED_FIXED_RULE_SUFFICES", "NOT_ESTABLISHED"],
        "jepa_feature_status": "NOT_IMPLEMENTED",
        "final_holdout_touched": False,
    }
    (root / "preregistration.json").write_text(json.dumps(prereg, indent=1) + "\n")
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit" / "PROTOCOL.locked.md").write_bytes((root / "PROTOCOL.md").read_bytes())
    (root / "audit" / "locked_at.txt").write_text(f"{prereg['locked_at']} {sha(root / 'preregistration.json')}\n")
    print("CAJU_LOCKED", prereg["locked_at"], sha(root / "preregistration.json"))


if __name__ == "__main__":
    main()
