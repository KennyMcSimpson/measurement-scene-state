"""Derives train_jepa_arkit_scale.py from train_jepa_arkit_norm.py (the JN script of
EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1) by exact replacements (each must occur exactly once). The only
change is the record source: arkit_records_scale (TRAIN plus four ended ARKit evaluation cohorts, 502 scans) instead of
arkit_records (TRAIN, 182 scans). Usage: python3 derive_train_scale.py SRC DST
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
REPLACEMENTS = [
    ('"""Accuracy-only reverse-JEPA training of DepthSplat on real sensor data (EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1, JN):\n',
     '"""Accuracy-only reverse-JEPA training on the scaled ARKit training set (EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1, JL):\n'
     'the JN script of EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1 with one change, the record source\n'
     '(arkit_records_scale: the 182 TRAIN scans plus four ended ARKit evaluation cohorts, 502 scans). JN was:\n'),
    ("from arkit_records import load_records  # noqa: E402\n",
     "from arkit_records_scale import load_records  # noqa: E402\n"),
    ('"source": "train_jepa_arkit_norm.py"', '"source": "train_jepa_arkit_scale.py"'),
]
for old, new in REPLACEMENTS:
    if text.count(old) != 1:
        raise SystemExit(f"replacement anchor found {text.count(old)} times: {old[:60]!r}")
    text = text.replace(old, new)
dst.write_text(text)
print("DERIVED", dst)
