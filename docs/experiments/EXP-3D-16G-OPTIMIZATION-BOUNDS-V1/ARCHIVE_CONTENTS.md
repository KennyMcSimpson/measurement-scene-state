# Portable numeric archive

The archive contains all numeric query/context rows, complete 153-chain trajectories (trace, full-objective checkpoints and all budget summaries), protocol, source snapshot, audits and the 782-state hash index. GPU weights, predictions and datasets stay in outputs.

From the repository root, reproduce primary statistics without media or weights:

`.venv/bin/python scripts/analyze_optimization_bounds.py --root /home/zonghan/measurement-scene-state/docs/experiments/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1 --output /tmp/optimization-bounds-primary-reproduction`

Add `--include-secondary` and use a different output directory for secondary analysis. The statistics loader reads compressed raw directly. Historical chronology is evidence of the original run; copying files does not recreate historical timestamps.
