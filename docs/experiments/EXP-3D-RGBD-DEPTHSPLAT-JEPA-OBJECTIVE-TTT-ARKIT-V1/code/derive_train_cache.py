"""Derives train_jepa_arkit_cache.py from train_jepa_arkit_norm.py (the accuracy-only script) by exact replacements
(each must occur exactly once). The only change is the episode layout of a training step, made the TTT evaluation's:
the state is built from the role's three context frames followed by the record's four Core B stream frames (a table
precomputed with the frozen stream_frame_ids rule, --streams) and rendered at the record's primary queries (frames 14
and 15). Usage: python3 derive_train_cache.py SRC DST
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
REPLACEMENTS = [
    ('"""Accuracy-only reverse-JEPA training of DepthSplat on real sensor data (EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1, JN):\n',
     '"""Seven-view accuracy-only reverse-JEPA training (EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1, JK): each step\n'
     'builds the state from the role\'s three context frames and the record\'s four Core B stream frames (--streams) and\n'
     'renders the record\'s primary queries (frames 14 and 15), the layout of the TTT evaluation; the objective is the\n'
     'accuracy-only one below. Derived from the JN script:\n'),
    ('    parser.add_argument("--min-hit", type=float, default=0.05)\n',
     '    parser.add_argument("--min-hit", type=float, default=0.05)\n'
     '    parser.add_argument("--streams", required=True)\n'),
    ('    table = json.loads(Path(args.hit_table).read_text())\n',
     '    table = json.loads(Path(args.hit_table).read_text())\n'
     '    streams = json.loads(Path(args.streams).read_text())["streams"]\n'),
    ('        while True:\n'
     '            record = rng.choice(records)\n'
     '            key = rng.choice(("context_a", "context_b"))\n'
     '            others = eligible[record["scene_id"]][key]\n'
     '            if len(others) >= args.targets:\n'
     '                break\n'
     '        context_ids = record["roles"][key]\n'
     '        if set(others) & set(context_ids):\n'
     '            raise ValueError("a context frame is listed as a target")\n'
     '        target_ids = rng.sample(others, args.targets)\n',
     '        record = rng.choice(records)\n'
     '        key = rng.choice(("context_a", "context_b"))\n'
     '        context_ids = list(record["roles"][key]) + list(streams[record["scene_id"]])\n'
     '        target_ids = list(record["roles"]["primary_query"])\n'
     '        if set(target_ids) & set(context_ids):\n'
     '            raise ValueError("a query frame would enter the state")\n'),
    ('"source": "train_jepa_arkit_norm.py"', '"source": "train_jepa_arkit_cache.py"'),
]
for old, new in REPLACEMENTS:
    if text.count(old) != 1:
        raise SystemExit(f"replacement anchor found {text.count(old)} times: {old[:60]!r}")
    text = text.replace(old, new)
dst.write_text(text)
print("DERIVED", dst)
