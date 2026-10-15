"""EXP-3D-CAJU-HEADROOM-V1 (pilot): prediction-level equivalence checks, no ground truth read (CPU).

1. Replay vs V1's independent runs on the first CAJU_PILOT episodes: OFF vs OFF_REF, SHORT vs SHORT_REF, FULL vs
   FULL_REF; the run-to-run floors OFF_REF vs OFF_REF2, SHORT_REF vs SHORT_REF2, FULL_REF vs FULL_REF2.
   Equivalent when max |difference| <= 2 x the floor + 1e-6 (depth_z in metres, alpha unitless).
2. Engine compatibility on CAJU_COMPAT: the replay's OFF against V3's sealed CACHE and FULL against V3's sealed
   TTT_m1 (released weights, cap 0.1, m = 1, 100 steps), with the floor of item 1 as the scale of GPU
   non-determinism.
Run through cv_host_run.py from the repository root (CUDA hidden):  python caju_equivalence_compare.py ROOT
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

V3_DS = Path("outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/ds/ARKIT_SYSTEM3/ZS").resolve()
ABS_TOL = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(item):
    if sha(item["path"]) != item["sha256"]:
        raise PermissionError(f"a sealed prediction changed: {item['path']}")
    return dict(np.load(item["path"]))


def merged(directory, pattern):
    predictions = {}
    for part in sorted(Path(directory).glob(pattern)):
        predictions.update(json.loads(part.read_text())["predictions"])
    return predictions


def max_diff(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compare(pairs, left, right):
    out = {}
    for name, (lk, rk) in pairs.items():
        out[name] = {f: max(max_diff(left[k][f"{lk}_{f}"], right[k][f"{rk}_{f}"]) for k in left)
                     for f in ("depth_z", "alpha")}
    return out


def main():
    root = Path(sys.argv[1]).resolve()
    replay = merged(root / "replay/PILOT", "replay_seal_*of*.json")
    reference = json.loads((root / "equivalence/equivalence_seal.json").read_text())["predictions"]
    keys = sorted(reference)
    ours = {k: load(replay[k]) for k in keys}
    refs = {k: load(reference[k]) for k in keys}
    pairs = {"OFF_vs_REF": ("OFF", "OFF_REF"), "SHORT_vs_REF": ("SHORT", "SHORT_REF"),
             "FULL_vs_REF": ("FULL", "FULL_REF")}
    floors = {"OFF_REF_vs_REF2": ("OFF_REF", "OFF_REF2"), "SHORT_REF_vs_REF2": ("SHORT_REF", "SHORT_REF2"),
              "FULL_REF_vs_REF2": ("FULL_REF", "FULL_REF2")}
    diffs = compare(pairs, {k: {**ours[k], **refs[k]} for k in keys}, {k: refs[k] for k in keys})
    floor = compare(floors, refs, refs)
    verdict = {}
    for action in ("OFF", "SHORT", "FULL"):
        d, f = diffs[f"{action}_vs_REF"], floor[f"{action}_REF_vs_REF2"]
        verdict[action] = all(d[x] <= 2 * f[x] + ABS_TOL for x in ("depth_z", "alpha"))
    v3 = merged(V3_DS, "ds_seal_*of*.json")
    compat = merged(root / "replay/COMPAT", "replay_seal_*of*.json")
    compat_keys = sorted(k for k, v in compat.items() if v["kind"] == "query")
    compat_ours = {k: load(compat[k]) for k in compat_keys}
    compat_v3 = {k: load(v3[k]) for k in compat_keys}
    compat_diffs = compare({"OFF_vs_V3_CACHE": ("OFF", "CACHE"), "FULL_vs_V3_TTT_m1": ("FULL", "TTT_m1")},
                           {k: {**compat_ours[k], **compat_v3[k]} for k in compat_keys}, compat_v3)
    report = {
        "pilot_keys": len(keys), "compat_keys": len(compat_keys), "tolerance": "max |diff| <= 2 x floor + 1e-6",
        "replay_vs_independent": diffs, "run_to_run_floor": floor, "equivalent": verdict,
        "compat_vs_v3": compat_diffs,
        "compat_within_floor": {
            "OFF": all(compat_diffs["OFF_vs_V3_CACHE"][x] <= 2 * floor["OFF_REF_vs_REF2"][x] + ABS_TOL
                       for x in ("depth_z", "alpha")),
            "FULL": all(compat_diffs["FULL_vs_V3_TTT_m1"][x] <= 2 * floor["FULL_REF_vs_REF2"][x] + ABS_TOL
                        for x in ("depth_z", "alpha"))},
        "ground_truth_read": False,
    }
    (root / "equivalence/equivalence_report.json").write_text(json.dumps(report, indent=1))
    print("CAJU_EQUIVALENCE", json.dumps({"equivalent": verdict, "compat": report["compat_within_floor"]}))


if __name__ == "__main__":
    main()
