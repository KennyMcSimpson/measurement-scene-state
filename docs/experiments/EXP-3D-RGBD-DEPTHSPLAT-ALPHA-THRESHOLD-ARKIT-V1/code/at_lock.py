"""EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1: preregistration.json and audit/locked_at.txt (refuses to overwrite a
lock or to lock once a selection or a score exists).
Run from the repository root:  python3 at_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1
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
    "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1", "EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1")}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
if (root / "preregistration.json").exists():
    raise SystemExit("already locked")
if any((root / n).exists() for n in ("selection.json", "scores", "results")):
    raise SystemExit("a selection or a score exists")
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {"reused_seals.json": sha(root / "reused_seals.json")},
    "earlier_preregistrations": {n: sha(p) for n, p in EARLIER.items()},
    "v1_code": {"score_analyze.py": sha(V1 / "code/score_analyze.py")},
    "design_basis": "the NF / NS results (known: the JN feed-forward gain is PARTIAL and saturated in data, coverage "
                    "0.72 -> ~0.55); locked before any threshold other than 0.5 was computed on any cohort",
    "settings": {"grid": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], "dev": ["ARKIT_SYSTEM", "ARKIT_SYSTEM2"],
                 "confirmation": ["ARKIT_SYSTEM3", "ARKIT_VAL"],
                 "selection": "tau_cache = argmin DEV scene mean of ZS_CACHE(t)|ALL; tau_core_b = argmin of "
                              "CORE_B(t)|ALL; ties: the smaller t", "jepa_readout": "JNS / JLS at t = 0.5"},
    "primary": {"P1_JEPA_VS_THRESHOLD_SYSTEM3_ALL": "ARKIT_SYSTEM3 ALL: ZS_CACHE(tau_cache) - JNS_CACHE",
                "P2_THRESHOLD_CORE_B_SYSTEM3_ALL": "ARKIT_SYSTEM3 ALL: CORE_B(0.5) - CORE_B(tau_core_b)"},
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
