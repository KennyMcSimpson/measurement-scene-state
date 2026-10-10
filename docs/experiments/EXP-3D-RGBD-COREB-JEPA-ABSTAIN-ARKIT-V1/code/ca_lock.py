"""EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a lock or to
lock once any seal, selection or score of this experiment exists).
Run from the repository root:  python3 ca_lock.py outputs/EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

OUT = Path("outputs").resolve()
V1 = OUT / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1"
EARLIER = {n: OUT / n / "preregistration.json" for n in (
    "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1", "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2", "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3",
    "EXP-3D-RGBD-ENSEMBLES-ARKIT-V1", "EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1",
    "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1", "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1",
    "EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1")}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
if (root / "preregistration.json").exists():
    raise SystemExit("already locked")
if any((root / n).exists() for n in ("ds", "selection.json", "scores", "results")):
    raise SystemExit("a seal, a selection or a score exists")
lock = {
    "experiment": "EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {n: sha(root / n) for n in ("reused_seals.json", "eval_SYSTEM/frame_role_lock.json",
                                              "eval_SYSTEM2/frame_role_lock.json")},
    "earlier_preregistrations": {n: sha(p) for n, p in EARLIER.items()},
    "v1_code": {n: sha(V1 / "code" / n) for n in ("ds_eval.py", "score_analyze.py")},
    "design_basis": "NF / NS / AT (known: JN's selective abstention improves the feed-forward state, a global threshold "
                    "does not) and the exploratory NF tie of JNS with CORE_B; locked before any candidate was computed",
    "settings": {"candidates": {"MASK": "sqrt(R7 x T'), T' = TTT ray where alpha_TTT >= 0.5 and alpha_JN >= 0.5, else R7",
                                "TRIPLE": "(R7 x ZS_TTT_F7 x JN_F7)^(1/3)", "PAIR": "sqrt(ZS_TTT_F7 x JN_F7)"},
                 "dev": ["ARKIT_SYSTEM", "ARKIT_SYSTEM2"], "confirmation": ["ARKIT_SYSTEM3", "ARKIT_VAL"],
                 "system_model": "JN1 (JN2 / JN3 secondary)", "dev_evaluation": "V1 ds_eval.py, JN1, CACHE arm, cap 0.1",
                 "selection": "lowest DEV scene mean ALL; ties MASK, TRIPLE, PAIR; stop unless below CORE_B"},
    "primary": {"P1_JEPA_ABSTAIN_VS_CORE_B_SYSTEM3_ALL": "ARKIT_SYSTEM3 ALL: CORE_B - SYS*",
                "P2_JEPA_ABSTAIN_VS_CORE_B_VAL_ALL": "ARKIT_VAL ALL: CORE_B - SYS*"},
    "statistics": {"unit": "scene", "draws": 10000, "seed": 20260928, "gate": "V2"},
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "the selection reads DEV query depth only and is hashed before any confirmation query depth is read",
              "ARKitScenes media outside git; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
