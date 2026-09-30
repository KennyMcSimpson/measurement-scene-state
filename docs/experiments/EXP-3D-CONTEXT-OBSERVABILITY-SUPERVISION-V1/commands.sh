#!/usr/bin/env bash
set -euo pipefail
# Run from repository root; full reruns require the original 17 exposed-scene data
# and the two sealed prior output directories. No protected holdout media needed.
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
PY=.venv/bin/python
REF=docs/experiments/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1
RUN=outputs/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1-reproduction
# Saved-raw numeric reproduction (no media or model loading).
"$PY" scripts/analyze_observability_supervision.py --root "$REF" --output /tmp/context-observability-reanalysis
# Fresh full rerun. Preserve the frozen protocol; never overwrite a previous run.
"$PY" scripts/initialize_observability_supervision_reproduction.py --reference "$REF" --output "$RUN"
"$PY" scripts/run_observability_supervision.py --root "$RUN" --phase baseline --device cuda
"$PY" scripts/evaluate_observability_supervision.py --root "$RUN" --phase baseline --device cuda
# A failed baseline reproduction exits nonzero and stops here.
"$PY" scripts/run_observability_supervision.py --root "$RUN" --phase formal --device cuda
"$PY" scripts/evaluate_observability_supervision.py --root "$RUN" --phase formal --device cuda
"$PY" scripts/audit_context_observability.py --root "$RUN"
"$PY" scripts/analyze_observability_supervision.py --root "$RUN"
.venv/bin/pytest -q
"$PY" scripts/audit_observability_supervision.py --root "$RUN"
# CUDA optimization does not promise identical floating-point tensors on rerun.
# Baseline tolerances, supervision weights and all thresholds remain frozen.
