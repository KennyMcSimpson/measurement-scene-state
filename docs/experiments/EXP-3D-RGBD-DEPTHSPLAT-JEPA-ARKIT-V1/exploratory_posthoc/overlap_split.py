"""EXPLORATORY (post hoc, after the locked analysis and amendment 1 of EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1): split the
ARKit FAR comparisons by source cohort (ARKIT_VAL, ARKIT_CONFIRM) and by context overlap. About half of the ARKit
queries have no context hit at all; there the harmonic is the Hypersim fill constant (2.37 m) everywhere. This
re-reads only the amendment-1 scene table (no prediction or depth is read again) and reports, per cohort, the V2
contrasts on FAR of the queries with at least one context hit and against the calibrated harmonic (H3CAL):
  H3 - HGN (the CONFIRM C1 contrast), H3CAL - HGN, H3 - JA, H3CAL - JA_CAL.
Run through cv_host_run.py from the repository root (CUDA hidden):  python overlap_split.py ROOT
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from jepa_arkit_score import contrast, write  # noqa: E402

PAIRS = {"H3-HGN|FAR_HIT_QUERIES": ("H3|FAR_HIT_QUERIES", "HGN|FAR_HIT_QUERIES"),
         "H3-HGN|FAR": ("H3|FAR", "HGN|FAR"), "H3CAL-HGN|FAR": ("H3CAL|FAR", "HGN|FAR"),
         "H3-JA|FAR_HIT_QUERIES": ("H3|FAR_HIT_QUERIES", "JA|FAR_HIT_QUERIES"),
         "H3CAL-JA_CAL|FAR": ("H3CAL|FAR", "JA_CAL|FAR")}

root = Path(sys.argv[1]).resolve()
table = json.loads((root / "amendment_1/scene_table.json").read_text())
rows = json.loads((root / "amendment_1/query_rows.json").read_text())
cohort_of = {r["scene_id"]: r["source_cohort"] for r in rows}
zero = {}
for r in rows:
    zero.setdefault(r["scene_id"], []).append(r["values"]["zero_hit"])
result = {"zero_hit_queries": {c: sum(sum(v) for s, v in zero.items() if cohort_of[s] == c) for c in set(cohort_of.values())},
          "scenes_all_queries_zero_hit": {c: sum(all(v) for s, v in zero.items() if cohort_of[s] == c)
                                          for c in set(cohort_of.values())},
          "contrasts": {}}
for cohort in ("ALL", "ARKIT_VAL", "ARKIT_CONFIRM"):
    keep = {s for s, c in cohort_of.items() if cohort in ("ALL", c)}
    sub = {k: {s: v for s, v in t.items() if s in keep} for k, t in table.items()}
    for name, (a, b) in PAIRS.items():
        s = contrast(sub, a, b)
        result["contrasts"][f"{cohort}:{name}"] = s
        print(f"{cohort:13s} {name:24s} {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
              f"{s['positive']}/{s['tied']}/{s['negative']} ({s['scenes']} scenes) {s['status']}")
print("zero-hit queries", result["zero_hit_queries"], "scenes with every query zero-hit", result["scenes_all_queries_zero_hit"])
write(root / "exploratory_posthoc/overlap_split.json", result)
