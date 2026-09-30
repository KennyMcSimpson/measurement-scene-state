#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root, with the archived source version and local data.
# The 17 exposed-scene paths in scene_manifest.json must exist. Protected holdout
# media is neither required nor permitted. These are attribution runs only.
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
PY=.venv/bin/python
REF=docs/experiments/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1
RUN=outputs/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1-reproduction

# Fast saved-raw reproduction: no media, checkpoints, or GPU access.
"$PY" scripts/analyze_optimization_bounds.py --root "$REF" --output /tmp/16g-primary-reanalysis
"$PY" scripts/analyze_optimization_bounds.py --root "$REF" --include-secondary --output /tmp/16g-secondary-reanalysis

# Full rerun: use a new output directory. Never refreeze budgets after scores.
"$PY" scripts/initialize_optimization_bounds_reproduction.py --reference "$REF" --output "$RUN"
"$PY" scripts/run_optimization_bounds.py --root "$RUN" --phase context --device cuda
"$PY" scripts/evaluate_optimization_bounds.py --root "$RUN" --phase context --device cuda
"$PY" scripts/run_optimization_bounds.py --root "$RUN" --phase oracle --device cuda
"$PY" scripts/evaluate_optimization_bounds.py --root "$RUN" --phase oracle --device cuda
"$PY" scripts/analyze_optimization_bounds.py --root "$RUN"
"$PY" scripts/run_optimization_bounds.py --root "$RUN" --phase secondary --device cuda
"$PY" scripts/evaluate_optimization_bounds.py --root "$RUN" --phase secondary --device cuda
"$PY" scripts/analyze_optimization_bounds.py --root "$RUN" --include-secondary --output "$RUN/secondary_analysis"
.venv/bin/pytest -q
"$PY" scripts/audit_optimization_bounds_statistics.py --root "$RUN"
"$PY" scripts/audit_optimization_bounds.py --root "$RUN"
# CUDA scatter accumulation is not bitwise deterministic. Raw-only statistics
# reproduce exactly; a new optimization run need not produce identical tensors.
# Optional phase wall times are measured separately; cumulative trajectory cost
# is always saved by the optimizer. Never sum shared budget prefixes as new runs.
