"""EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1: training/hit_shares_SCALE.json = the TRAIN hit table of
OBJECTIVE-TTT (unchanged) merged with the tables of the four added cohorts (OVERLAP-V1's hit_share_table.py: context
depth and target cameras only). Keys must be disjoint and must cover exactly the training records of
arkit_records_scale. Run from the repository root:  python3 merge_hit_tables.py OUT TRAIN.json ADDED.json [ADDED.json ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arkit_records_scale import load_records  # noqa: E402

out, sources = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
merged = {}
for path in sources:
    table = json.loads(path.read_text())
    if set(table) & set(merged):
        raise ValueError(f"{path}: scene ids overlap an earlier table")
    merged.update(table)
train, _ = load_records()
if set(merged) != {r["scene_id"] for r in train}:
    raise ValueError("the merged table does not cover exactly the training records")
if any(set(roles) != {"A", "B"} for roles in merged.values()):
    raise ValueError("every record needs both context roles")
out.write_text(json.dumps(merged, indent=0))
print("MERGED", len(merged), "records from", [p.name for p in sources])
