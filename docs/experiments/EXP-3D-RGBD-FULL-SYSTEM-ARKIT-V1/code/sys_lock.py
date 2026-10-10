"""EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a lock, to lock
after any prediction exists, or when PROTOCOL.md or the cohort script differ from the plan lock written before the
download). Run from the repository root:  python3 sys_lock.py outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

DATA = Path("outputs/EXP-3D-RGBD-ARKIT-SYSTEM-DATA-V1").resolve()
CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1").resolve()
V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()
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
if (root / "raw").exists() or (root / "ds").exists() or (root / "geometry").exists() or (root / "scores").exists():
    raise SystemExit("a prediction or score exists")
plan = (root / "audit/plan_locked_at.txt").read_text()
script = Path("scripts/prepare_rgbd_arkit_system_cohort.py")
if f"sha256(PROTOCOL.md)={sha(root / 'PROTOCOL.md')}" not in plan:
    raise SystemExit("PROTOCOL.md differs from the plan lock")
if f"sha256(scripts/prepare_rgbd_arkit_system_cohort.py)={sha(script)}" not in plan:
    raise SystemExit("the cohort script differs from the plan lock")
lock = {
    "experiment": "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "plan_lock": plan.strip(),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "evaluation_contract.json", "selector_seal.json")},
    "cohort_data": {n: sha(DATA / n) for n in ("selection.json", "downloads.json", "candidate_lock.json",
                                               "manifest_arkit_system.json", "preparation_integrity.json")},
    "cohort_script_sha256": sha(script),
    "models": {"GA_contract_sha256": sha(CARRIER / "evaluation_contract.json"), "released_weights_sha256": sha(RELEASED)},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "geometry_seal.py", "score_analyze.py")},
    "settings": {"core_b": "sqrt(R7 x ZS_TTT_F7), ZS = released weights at cap 0.1, TTT m = 1, 100 steps",
                 "full": "Core B with R7 replaced on FAR7 pixels of episodes with S >= 1.2 by GAN7 (GA normalized, "
                         "R7 fallback)", "alpha_min": 0.5, "opacity_min": 0.5},
    "primary": {"P1_FULL_SYSTEM_VS_CORE_B_ALL": "ALL: CORE_B - FULL", "P2_FULL_SYSTEM_VS_MODEL_FREE_ALL": "ALL: R7 - FULL"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by sys_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit/locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                          f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
