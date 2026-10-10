"""EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1: preregistration.json (sha256 of every input of the formal run) and
audit/locked_at.txt; refuses to overwrite a lock. Run from the repository root:
  python3 ablation_lock.py outputs/EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

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
selection = json.loads((root / "nv_selection.json").read_text())
lock = {
    "experiment": "EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1",
    "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "code": {p.name: sha(p) for p in sorted((root / "code").iterdir()) if p.is_file()},
    "root_files": {name: sha(root / name) for name in ("frame_role_lock.json", "nv_selection.json",
                                                        "geometry/geometry_seal_FRESH78.json")},
    "v1": {"preregistration_sha256": sha(V1 / "preregistration.json"),
           "arms_seal_sha256": sha(V1 / "raw/eval/FRESH78/prediction_seal.json"),
           "zs_seals": {p.name: sha(p) for p in sorted((V1 / "ds/FRESH78/ZS").glob("ds_seal_*of*.json"))}},
    "settings": {"nv_scale_c": selection["scale_c"], "nv_opacity": 0.95, "nv_ttt_mult": selection["ttt_mult"],
                 "ttt_steps": 100, "zs_ttt_mult": 1.0, "alpha_min": 0.5,
                 "bootstrap": {"draws": 10000, "seed": 20260928, "unit": "scene"}},
    "primary": {"P1_LEARNED_BASE_UNDER_TTT_ALL": "ALL: NV_TTT_F7 - ZS_TTT_F7",
                "P2_LEARNED_BASE_CACHE_ALL": "ALL: NV_CACHE_F7 - ZS_CACHE_F7"},
    "branch": "LEARNED_BASE_MATTERS if P1 passes the V2 gate; NAIVE_BETTER if P1's CI high < 0; TTT_IS_GENERIC if "
              "P1's CI spans 0 and P3 (NV_CACHE_F7 - NV_TTT_F7) passes; else INCONCLUSIVE",
    "rules": ["no setting changes after this lock; amendments only appended and dated",
              "query depth read only by the V1 score stage after the NV seals are complete",
              "protected holdout never opened; nothing pushed"],
}
(root / "preregistration.json").write_text(json.dumps(lock, indent=1) + "\n")
(root / "audit").mkdir(exist_ok=True)
(root / "audit" / "locked_at.txt").write_text(f"{lock['locked_at']} sha256(preregistration.json)="
                                              f"{sha(root / 'preregistration.json')}\n")
print("LOCKED", lock["locked_at"], sha(root / "preregistration.json"))
