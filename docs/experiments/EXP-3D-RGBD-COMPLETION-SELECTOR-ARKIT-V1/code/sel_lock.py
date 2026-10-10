"""EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a lock,
to lock after any prediction exists, or when PROTOCOL.md differs from the plan lock written before the download).
Run from the repository root:  python3 sel_lock.py outputs/EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

DATA = Path("outputs/EXP-3D-RGBD-ARKIT-SELECT-DATA-V1").resolve()
CARRIER = Path("outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1").resolve()
OVERLAP = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1").resolve()
V1 = Path("outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1").resolve()


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
if (root / "preregistration.json").exists():
    raise SystemExit("already locked")
if (root / "raw").exists() or (root / "ds").exists() or (root / "scores").exists():
    raise SystemExit("a prediction or score exists")
plan = (root / "audit/plan_locked_at.txt").read_text()
if f"sha256(PROTOCOL.md)={sha(root / 'PROTOCOL.md')}" not in plan:
    raise SystemExit("PROTOCOL.md differs from the plan lock")
script = Path("scripts/prepare_rgbd_arkit_select_cohort.py")
if f"sha256(scripts/prepare_rgbd_arkit_select_cohort.py)={sha(script)}" not in plan:
    raise SystemExit("the cohort script differs from the plan lock")
lock = {
    "experiment": "EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "plan_lock": plan.strip(),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("frame_role_lock.json", "evaluation_contract.json",
                                              "manifest_arkit_select_overlap.json", "selector_seal.json")},
    "cohort_data": {n: sha(DATA / n) for n in ("selection.json", "downloads.json", "candidate_lock.json",
                                               "manifest_arkit_select.json", "preparation_integrity.json")},
    "cohort_script_sha256": sha(script),
    "models": {"GA_contract_sha256": sha(CARRIER / "evaluation_contract.json"),
               "JO_checkpoints": {f"JO{s}": sha(OVERLAP / f"checkpoints/JO{s}/depthsplat_jepa_final.pth")
                                  for s in (1, 2, 3)}},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "settings": {"selector": "S = (p90 - p10) / median of the episode's valid context depth; learned where S >= 1.2",
                 "disagreement_selector": "GAN where median |log(GAN / H3)| outside NEAR >= 0.18",
                 "opacity_min": 0.5, "alpha_min": 0.5, "region": "FARH"},
    "primary": {"P1_SELECTED_CARRIER_VS_HARMONIC_FARH": "FARH: H3 - SEL_GAN"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by sel_score.py score after every seal is complete",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit/locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                          f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
