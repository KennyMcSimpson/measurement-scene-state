# Portable numeric archive

The archive contains all numeric query/context rows, complete 136-state trajectories (trace, full-objective checkpoints and fixed-final summaries), protocol, source snapshot, audits and the 136-state hash index. GPU weights, predictions and datasets stay in outputs.

From the repository root, reproduce primary statistics without media or weights:

`.venv/bin/python scripts/analyze_observability_supervision.py --root /home/zonghan/measurement-scene-state/docs/experiments/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1 --output /tmp/context-geometry-statistics-reproduction`

The statistics loader reads compressed raw directly. Historical chronology is evidence of the original run; copying files does not recreate historical timestamps.
