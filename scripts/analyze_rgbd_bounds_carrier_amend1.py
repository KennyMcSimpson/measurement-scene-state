#!/usr/bin/env python3
"""V7 Amendment 1: the frozen V7 analysis with one inapplicable V2-estimator assertion removed.

The frozen V2 estimator asserts that C0 and C1 share per-scene geometric ray-hit fractions, a
V2-V6 premise (one bounds rule for every variant). V7's factor is the bounds rule, so the
fractions differ by design. This entry removes exactly that two-line assertion from the frozen
estimator source (and checks that nothing else changes), runs the frozen V7 analysis script with
it, and also writes the per-variant ray-hit fractions (descriptive). See AMENDMENTS.md.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

from mcss.mechanism_pilot import geometry_carrier_statistics as frozen

CHECK = (
    '    if any(abs(ray_hits["C0"][s] - ray_hits["C1"][s]) > 1e-8 for s in ids):\n'
    '        raise ValueError("Matched variants must share frozen geometry ray-hit fractions")\n'
)
FROZEN_ANALYSIS = Path(__file__).with_name("analyze_rgbd_bounds_carrier.py")
AMENDMENT = (
    "V7 Amendment 1 (AMENDMENTS.md): the frozen V2 estimator without its inapplicable "
    "C0/C1 ray-hit-fraction equality assertion; everything else unchanged"
)


def amended_analyze():
    """frozen.analyze minus the two-line equality assertion, with the frozen module globals."""
    source = inspect.getsource(frozen.analyze)
    if source.count(CHECK) != 1:
        raise RuntimeError("Frozen estimator must contain the ray-hit assertion exactly once")
    namespace = dict(vars(frozen))
    exec(compile(source.replace(CHECK, ""), frozen.__file__, "exec"), namespace)
    return namespace["analyze"]


def frozen_analysis_module():
    spec = importlib.util.spec_from_file_location("rgbd_bounds_frozen_analysis", FROZEN_ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.analyze = amended_analyze()
    return module


def run(root, output=None):
    root = Path(root)
    output = Path(output) if output else root
    module = frozen_analysis_module()
    artifacts = module.run(root, output)
    rows = module.load(root, "raw/dev_query_results.json", {})
    fractions = {
        v: frozen._macro(
            [r for r in rows if r["variant"] == v and r["method"] == "direct"], "ray_hitfraction"
        )
        for v in sorted({r["variant"] for r in rows})
    }
    module.save(
        output / "ray_hit_fractions.json",
        {
            "descriptive_only": True,
            "definition": "per-scene equal mean (roles, then queries) of the fraction of query "
            "rays that intersect the state's bounds box, direct states",
            "per_variant_scene": fractions,
            "amendment": AMENDMENT,
        },
    )
    record = output / "statistics_reproduction.json"
    value = json.loads(record.read_text())
    value["amendment"] = AMENDMENT
    value["source_sha256"][str(Path(__file__).resolve())] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    module.save(record, value)
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
