"""EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1 development selection on DEV (before the lock): from a DEV scene table,
the readout family with the lowest mean ALL AbsRel; ties go to the earlier candidate.
  python ablation_select.py SCENE_TABLE OUT KEY=READOUT|ALL [KEY=READOUT|ALL ...]
"""
import json
import sys

import numpy as np

table = json.load(open(sys.argv[1]))
candidates = [spec.split("=", 1) for spec in sys.argv[3:]]
means = {key: float(np.mean(list(table[name].values()))) for key, name in candidates}
best = min(means, key=lambda k: (round(means[k], 12), [c[0] for c in candidates].index(k)))
json.dump({"means": means, "readouts": dict(candidates), "selected": best}, open(sys.argv[2], "w"), indent=1)
print("SELECTED", best, json.dumps(means))
