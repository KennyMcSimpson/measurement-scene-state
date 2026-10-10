"""Derives train_jepa_arkit_overlap_scale.py from EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1/code/train_jepa_arkit_overlap.py
by exact replacements (each must occur exactly once). The only change is the record source: arkit_records_scale (the 182
TRAIN scans plus four ended ARKit evaluation cohorts, 502 scans) instead of arkit_records.
Usage: python3 derive_train_overlap_scale.py SRC DST
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
REPLACEMENTS = [
    ('"""Overlap-aware reverse-JEPA training of DepthSplat on real sensor data (EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1):\n',
     '"""Overlap-aware reverse-JEPA training on the scaled ARKit training set (EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1,\n'
     'JOL): the OVERLAP-V1 script with one change, the record source (arkit_records_scale: the 182 TRAIN scans plus four\n'
     'ended ARKit evaluation cohorts, 502 scans). OVERLAP-V1 was:\n'),
    ("from arkit_records import load_records  # noqa: E402\n",
     "from arkit_records_scale import load_records  # noqa: E402\n"),
    ('"source": "train_jepa_arkit_overlap.py"', '"source": "train_jepa_arkit_overlap_scale.py"'),
]
for old, new in REPLACEMENTS:
    if text.count(old) != 1:
        raise SystemExit(f"replacement anchor found {text.count(old)} times: {old[:60]!r}")
    text = text.replace(old, new)
dst.write_text(text)
print("DERIVED", dst)
