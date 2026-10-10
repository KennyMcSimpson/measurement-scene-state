"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1: record amendment 1 (refuses once any ARKIT160 query depth may have been
read, i.e. once scores/ exists, and refuses to overwrite). Appends the dated text to PROTOCOL.md (the locked text
stays an exact prefix) and writes audit/amendment_1.json.
Run from the repository root:  python3 amend1_record.py ROOT TEXT.md
"""
import hashlib
import json
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root, text_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
record_path = root / "audit/amendment_1.json"
if record_path.exists():
    raise SystemExit("amendment 1 already recorded")
if (root / "scores").exists() or (root / "amendment_1/scene_table.json").exists():
    raise SystemExit("scoring has started: too late for an amendment")
prereg = json.loads((root / "preregistration.json").read_text())
protocol = root / "PROTOCOL.md"
before = protocol.read_bytes()
if hashlib.sha256(before).hexdigest() != prereg["protocol_sha256"]:
    raise SystemExit("PROTOCOL.md differs from the locked text")
calibration = json.loads((root / "amendment_1/h3_calibration_TRAIN.json").read_text())
now = time.strftime("%Y-%m-%d %H:%M:%S %z")
protocol.write_bytes(before + text_path.read_text().replace("{TIME}", now).encode())
files = sorted(p for p in (root / "amendment_1").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
record = {
    "amendment": 1, "recorded_at": now,
    "before_any_arkit160_query_depth_read": True,
    "reason": "on ARKit TRAIN, 50% of queries have no context hit; the harmonic is then the Hypersim fill constant "
              "2.37 m everywhere (FAR AbsRel 1.05 vs 0.38 with hits); a TRAIN-fitted calibrated harmonic separates "
              "this constant from completion information",
    "k_hit": calibration["k_hit"], "c_zero_hit_m": calibration["c_zero_hit_m"],
    "calibration_sha256": sha(root / "amendment_1/h3_calibration_TRAIN.json"),
    "files": {str(p.relative_to(root)): sha(p) for p in files},
    "contrasts": ["X1 FAR: H3CAL - JA_CAL", "X2 FAR: H3 - H3CAL", "X3 FAR: H3CAL - JA", "X4 FAR: H3CAL - HGN",
                  "X5 JA-covered FAR: H3CAL - JA (descriptive)", "X6 FAR of queries with hits: H3 - JA (descriptive)"],
    "unchanged": "R1, R2, S1-S9, the locked code, PROTOCOL prefix and every seal",
    "protocol_sha256_locked": prereg["protocol_sha256"], "protocol_sha256_after": sha(protocol),
}
record_path.write_text(json.dumps(record, indent=1) + "\n")
print("AMENDMENT_1_RECORDED", now, "k_hit", record["k_hit"], "c_zero_hit_m", record["c_zero_hit_m"],
      "sha", sha(record_path))
