"""Summarize DepthSplat probe variants from their scores.json files (exploratory, descriptive).

For each variant directory under BASE: per-scene means (over roles and queries; G1 also over seeds) of AbsRel per
method and region, FAR coverage (alpha >= 0.5), and paired scene differences a - b (positive = b better) with a
scene bootstrap 95% interval (10,000 draws, seed 20260928) and better/worse scene counts. Scenes without pixels in a
region (NaN) are left out of that region's statistics.
  FAR_DSN_SCALED  harmonic filling in NEAR and in uncovered FAR, DepthSplat's normalized depth in covered FAR
  FAR_G1          harmonic filling in NEAR, G1 in FAR (the project's AHFILL8-style hybrid)
  FAR_DSN_G1      harmonic filling in NEAR, DepthSplat in covered FAR, G1 in uncovered FAR
Usage (project environment): python ds_summary.py BASE OUT_JSON variant_dir ...
"""

import json
import sys
from pathlib import Path

import numpy as np

REGIONS = ("ALL", "NEAR", "FAR")
METHODS = ("REPROJ_HARMONIC", "G1", "DS", "DSN_OR_HARM_SCALED", "FAR_DSN_SCALED", "FAR_G1", "FAR_DSN_G1")
CONTRASTS = (("REPROJ_HARMONIC", "FAR_DSN_SCALED"), ("REPROJ_HARMONIC", "FAR_G1"), ("FAR_G1", "FAR_DSN_G1"),
             ("FAR_DSN_SCALED", "FAR_DSN_G1"))


def scene_means(rows, method, region, scenes):
    out = []
    for s in scenes:
        values = [r[region] for r in rows if r["scene_id"] == s and r["method"] == method and r[region] is not None]
        out.append(float(np.mean(values)) if values else np.nan)
    return np.array(out)


def bootstrap(d, draws=10_000, seed=20260928):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def contrast(rows, a, b, region, scenes):
    d = scene_means(rows, a, region, scenes) - scene_means(rows, b, region, scenes)
    d = d[np.isfinite(d)]
    lo, hi = bootstrap(d)
    return {"mean": float(d.mean()), "ci95": [lo, hi], "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "scenes": int(d.size)}


def fmt(c):
    return f"{c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] {c['better']}/{c['worse']}"


def main():
    base, out_json, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    summary = {}
    for name in names:
        scores = json.loads((base / name / "scores.json").read_text())
        rows, coverage = scores["rows"], scores["coverage"]
        scenes = sorted({r["scene_id"] for r in rows})
        present = {r["method"] for r in rows}
        far_cov = np.nanmean([np.mean([c["far_covered"] for c in coverage
                                       if c["scene_id"] == s and c["far_covered"] is not None] or [np.nan])
                              for s in scenes])
        entry = {"scenes": len(scenes), "far_covered": float(far_cov), "methods": {}, "contrasts": {}}
        for method in METHODS:
            if method in present:
                entry["methods"][method] = {r: float(np.nanmean(scene_means(rows, method, r, scenes)))
                                            for r in REGIONS}
        for a, b in CONTRASTS:
            if a in present and b in present:
                for region in ("FAR", "ALL"):
                    entry["contrasts"][f"{a} - {b} [{region}]"] = contrast(rows, a, b, region, scenes)
        summary[name] = entry
        m, c = entry["methods"], entry["contrasts"]
        line = (f"| {name} | {far_cov:.2f} | {m['FAR_DSN_SCALED']['FAR']:.4f} | "
                f"{fmt(c['REPROJ_HARMONIC - FAR_DSN_SCALED [FAR]'])} |")
        if "FAR_DSN_G1" in m:
            line += (f" {m['FAR_G1']['FAR']:.4f} | {m['FAR_DSN_G1']['FAR']:.4f} | "
                     f"{fmt(c['FAR_G1 - FAR_DSN_G1 [FAR]'])} | {fmt(c['FAR_G1 - FAR_DSN_G1 [ALL]'])} |")
        print(line)
    out_json.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
