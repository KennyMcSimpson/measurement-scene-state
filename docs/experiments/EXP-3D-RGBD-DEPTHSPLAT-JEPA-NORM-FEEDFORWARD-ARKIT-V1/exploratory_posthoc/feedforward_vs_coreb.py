"""EXPLORATORY (post hoc, after both results were known): on ARKIT_SYSTEM3, per-scene comparison of the JN feed-forward
state (the NF scene table) with Core B and TTT alone (the scene table of EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3). Both tables
use the same frame-role lock, geometry seal, F7 readout and scene aggregation. Not preregistered; labels only.
Run through cv_host_run.py from the repository root (CUDA hidden):  python feedforward_vs_coreb.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent / "EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3"
sys.path.insert(0, str(ROOT.parent / "EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1" / "code"))
from score_analyze import contrast, write  # noqa: E402

if (ROOT / "frame_role_lock.json").read_bytes() != (V3 / "frame_role_lock.json").read_bytes():
    raise SystemExit("different frame-role locks")
nf = json.loads((ROOT / "scores/scene_table.json").read_text())
v3 = json.loads((V3 / "scores/scene_table.json").read_text())
table = {"CORE_B|ALL": v3["CORE_B|ALL"], "ZS_TTT_F7|ALL": v3["ZS_TTT_F7|ALL"],
         "JNS_CACHE|ALL": nf["JNS_CACHE|ALL"], "JN1_CACHE|ALL": nf["JN1_CACHE|ALL"]}
same_r7 = max(abs(v3["R7|ALL"][s] - nf["R7|ALL"][s]) for s in nf["R7|ALL"])
out = {"label": "exploratory post hoc, not preregistered", "r7_max_abs_diff_between_experiments": same_r7}
for name, (a, b) in {"CORE_B_minus_JNS": ("CORE_B|ALL", "JNS_CACHE|ALL"),
                     "CORE_B_minus_JN1": ("CORE_B|ALL", "JN1_CACHE|ALL"),
                     "TTT_minus_JNS": ("ZS_TTT_F7|ALL", "JNS_CACHE|ALL")}.items():
    s = contrast(table, a, b)
    out[name] = {k: s[k] for k in ("mean", "ci95", "positive", "tied", "negative", "median", "status")}
    print(name, f"{s['mean']:+.4f}", [round(x, 4) for x in s["ci95"]], f"{s['positive']}/{s['tied']}/{s['negative']}",
          f"median {s['median']:+.4f}", s["status"])
print("R7 max abs diff between the two experiments:", same_r7)
write(ROOT / "exploratory_posthoc/feedforward_vs_coreb.json", out)
