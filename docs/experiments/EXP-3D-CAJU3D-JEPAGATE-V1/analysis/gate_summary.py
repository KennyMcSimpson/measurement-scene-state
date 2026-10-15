"""EXP-3D-CAJU3D-JEPAGATE-V1 descriptive DEV summary (protocol section 4; after dev-score, reads only the written
DEV results, the DEV score rows and the sealed DEV logs): means, every pairwise place contrast (Core B readout and
F7 alone), dropped-set overlaps, TTT depth terms, score summaries, query-view coverage.
Run from the repository root (CUDA hidden):  python gate_summary.py ROOT > ROOT/results/dev_summary.txt
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
OUTPUTS = ROOT.parent
for _path in ("EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code", "EXP-3D-CAJU-HEADROOM-V1/code", "EXP-3D-CAJU3D-ENGINE-V1/code"):
    sys.path.insert(0, str(OUTPUTS / _path))
import engine_score as S  # noqa: E402
from score_analyze import contrast  # noqa: E402

ARMS = ("JEPA", "DEPTH", "INC", "RANDOM")
NAMES = ("CORE_B", "ZS_CACHE") + tuple(f"GATE_{a}" for a in ARMS)


def line(c):
    i, t, w = c["improved_tied_worse"]
    return (f"{c['mean']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] {i}/{t}/{w} {c['status']}"
            f" loso [{c['loso_range'][0]:+.5f}, {c['loso_range'][1]:+.5f}]")


res = json.loads((ROOT / "results/dev_results.json").read_text())
rows = json.loads((ROOT / "scores/DEV/rows.json").read_text())
logs = {}
for part in sorted((ROOT / "gate/DEV").glob("gate_seal_*of*.json")):
    logs.update(json.loads(part.read_text())["episodes_log"])
print(f"DEV places {res['places']}, episodes {len(logs)}, decision {'GO' if res['go'] else 'STOP'}")
for metric in ("core_b", "f7"):
    table = S.place_table(rows, metric)
    print(f"\n[{metric}] place means (lower is better)")
    for name in sorted(table, key=lambda n: S.mean(table[n])):
        print(f"  {name:12s} {S.mean(table[name]):.5f}")
    print(f"[{metric}] pairwise contrasts left - right (> 0: right is better)")
    for left, right in itertools.combinations(NAMES, 2):
        print(f"  {left:12s} - {right:12s} {line(contrast(table, left, right))}")
coverage = {n: float(np.mean([r["coverage"][n] for r in rows])) for n in NAMES}
print("\nquery-view coverage (alpha >= 0.5 on valid ground truth):",
      json.dumps({k: round(v, 4) for k, v in coverage.items()}))
print("dropped-set overlap (share of the first arm's dropped pixels also dropped by the second; chance 0.10):")
print("  ", json.dumps({k: round(v, 3) for k, v in res["dropped_overlap"].items()}))
print("TTT depth term on the kept pixels (first step -> last step), failures, seconds per arm:")
for arm in ARMS:
    a = [v["arms"][arm] for v in logs.values()]
    print(f"  {arm:7s} {np.mean([x['depth_term_first'] for x in a]):.4f} -> {np.mean([x['depth_term_last'] for x in a]):.4f}"
          f"  failures {sum(x['failed_at_step'] is not None for x in a)}  kept {np.mean([x['kept_valid_share'] for x in a]):.3f}"
          f"  {np.mean([x['seconds'] for x in a]):.1f} s")
print("score summary over episodes (mean of the episode mean, mean of the episode max):")
for arm in ARMS:
    s = [v["arms"]["score_summary"][arm] for v in logs.values()]
    print(f"  {arm:7s} {np.mean([x[0] for x in s]):.4f} {np.mean([x[1] for x in s]):.4f}")
