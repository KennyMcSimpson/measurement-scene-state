"""Derives train_jepa_arkit_overlap.py from EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1/code/train_jepa_arkit.py by exact
replacements (each must occur exactly once): the only change is that training targets are drawn among the record's
non-context frames whose context hit share (hit table, frozen geometry) is at least --min-hit (default 0.05); a
record/role with fewer eligible frames than --targets is redrawn. Usage: python3 derive_train_overlap.py SRC DST
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
REPLACEMENTS = [
    ('"""Reverse-JEPA training of DepthSplat on real sensor data (EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1): the recipe of\n'
     'EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/train_jepa.py with only the records changed: ARKitScenes TRAIN and DEV\n'
     '(arkit_records.load_records) instead of Hypersim TRAIN72 + SEG-V1 and the Hypersim DEV.\n',
     '"""Overlap-aware reverse-JEPA training of DepthSplat on real sensor data (EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1):\n'
     'EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1/code/train_jepa_arkit.py with one change: the targets of a step are drawn\n'
     'only among the non-context frames that the role\'s three context frames hit on at least --min-hit of their pixels\n'
     '(hit table from the frozen 1-pixel z-buffer reprojection, computed without target depth); a record/role with\n'
     'fewer eligible frames than --targets is redrawn. On ARKit about a third of the non-context frames share no pixel\n'
     'with the context, so the original recipe spends much of its signal on views that cannot be completed.\n'),
    ('  python train_jepa_arkit.py OUT --seed S [--steps 3000] [--lr 5e-6] [--targets 2] [--scale-max 3.0]\n',
     '  python train_jepa_arkit_overlap.py OUT --seed S --hit-table HIT.json [--min-hit 0.05] [--steps 3000] [--lr 5e-6]\n'
     '                                     [--targets 2] [--scale-max 3.0]\n'),
    ('  python train_jepa_arkit.py OUT --eval-only',
     '  python train_jepa_arkit_overlap.py OUT --hit-table HIT.json --eval-only'),
    ('    parser.add_argument("--no-save", action="store_true", help="smoke runs: skip the final checkpoint")\n',
     '    parser.add_argument("--no-save", action="store_true", help="smoke runs: skip the final checkpoint")\n'
     '    parser.add_argument("--hit-table", required=True)\n'
     '    parser.add_argument("--min-hit", type=float, default=0.05)\n'),
    ('    records, dev = load_records()\n',
     '    records, dev = load_records()\n'
     '    table = json.loads(Path(args.hit_table).read_text())\n'
     '    eligible = {r["scene_id"]: {key: sorted(int(f) for f, share in table[r["scene_id"]][role].items()\n'
     '                                            if share >= args.min_hit)\n'
     '                                for role, key in (("A", "context_a"), ("B", "context_b"))} for r in records}\n'),
    ('    log = {"args": vars(args), "overrides": list(ds_predict.OVERRIDES), "train_records": len(records),\n',
     '    log = {"args": vars(args), "overrides": list(ds_predict.OVERRIDES), "train_records": len(records),\n'
     '           "eligible_targets": sum(len(v) for roles in eligible.values() for v in roles.values()),\n'
     '           "eligible_roles": sum(len(v) >= args.targets for roles in eligible.values() for v in roles.values()),\n'),
    ('        record = rng.choice(records)\n'
     '        key = rng.choice(("context_a", "context_b"))\n'
     '        context_ids = record["roles"][key]\n'
     '        others = [f["frame_id"] for f in record["frames"] if f["frame_id"] not in context_ids]\n'
     '        target_ids = rng.sample(others, args.targets)\n',
     '        while True:\n'
     '            record = rng.choice(records)\n'
     '            key = rng.choice(("context_a", "context_b"))\n'
     '            others = eligible[record["scene_id"]][key]\n'
     '            if len(others) >= args.targets:\n'
     '                break\n'
     '        context_ids = record["roles"][key]\n'
     '        if set(others) & set(context_ids):\n'
     '            raise ValueError("a context frame is listed as a target")\n'
     '        target_ids = rng.sample(others, args.targets)\n'),
    ('"source": "train_jepa_arkit.py"', '"source": "train_jepa_arkit_overlap.py"'),
]
for old, new in REPLACEMENTS:
    if text.count(old) != 1:
        raise SystemExit(f"replacement anchor found {text.count(old)} times: {old[:60]!r}")
    text = text.replace(old, new)
dst.write_text(text)
print("DERIVED", dst)
