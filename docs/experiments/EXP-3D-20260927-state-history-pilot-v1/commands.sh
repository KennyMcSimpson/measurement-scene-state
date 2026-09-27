#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/pytest -q
.venv/bin/ruff check src/mcss/mechanism_pilot scripts/run_3d_mechanism_smoke.py scripts/finalize_3d_mechanism_pilot.py tests/test_mechanism*.py
# New directory mandatory: refuses overwrite. Synthetic engineering, no scientific evaluation.
.venv/bin/python scripts/run_3d_mechanism_smoke.py --output outputs/EXP-3D-20260927-state-history-pilot-v1/smoke_reproduction
# Recompute report from existing records; never loads real GT or modifies sealed V1/V2.
.venv/bin/python scripts/finalize_3d_mechanism_pilot.py
