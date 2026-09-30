#!/usr/bin/env bash
# Run from the measurement-scene-state repository root.
set -euo pipefail
MCSS_PYTHON=${MCSS_PYTHON:-.venv/bin/python}
REFERENCE_ROOT=outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1
case "${1:-help}" in
  reproduce)
    CAPACITY_RUN=${2:?Specify a NEW output directory}
    "$MCSS_PYTHON" scripts/initialize_direct_capacity_reproduction.py --reference "$REFERENCE_ROOT" --output "$CAPACITY_RUN"
    PYTHONFAULTHANDLER=1 "$MCSS_PYTHON" scripts/run_direct_capacity_optimization.py --root "$CAPACITY_RUN" --phase context --device cuda
    "$MCSS_PYTHON" scripts/evaluate_direct_capacity.py --root "$CAPACITY_RUN" --phase context --device cuda
    PYTHONFAULTHANDLER=1 "$MCSS_PYTHON" scripts/run_direct_capacity_optimization.py --root "$CAPACITY_RUN" --phase oracle --device cuda
    "$MCSS_PYTHON" scripts/evaluate_direct_capacity.py --root "$CAPACITY_RUN" --phase oracle --device cuda
    "$MCSS_PYTHON" scripts/analyze_direct_capacity.py --root "$CAPACITY_RUN"
    "$MCSS_PYTHON" scripts/audit_direct_capacity.py --root "$CAPACITY_RUN"
    ;;
  analyze)
    CAPACITY_ANALYSIS=${2:?Specify an output directory for regenerated analysis}
    "$MCSS_PYTHON" scripts/analyze_direct_capacity.py --root "$REFERENCE_ROOT" --output "$CAPACITY_ANALYSIS"
    ;;
  tests)
    "$MCSS_PYTHON" -m pytest -q
    ;;
  *)
    echo 'Usage: bash outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1/commands.sh reproduce NEW_OUTPUT_DIR | analyze ANALYSIS_DIR | tests'
    echo 'Requires archived data paths and the exact frozen source versions. Reproduce reuses the historical baseline and context-only pilot protocol, then fits all 425 new states.'
    ;;
esac
