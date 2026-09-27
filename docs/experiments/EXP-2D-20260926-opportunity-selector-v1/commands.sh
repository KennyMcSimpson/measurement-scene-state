#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
PY=.venv/bin/python
$PY scripts/run_opportunity_probe.py prepare
$PY scripts/run_opportunity_probe.py discovery
$PY scripts/run_opportunity_probe.py lock
$PY scripts/run_opportunity_probe.py validation
$PY scripts/run_opportunity_probe.py report

$PY scripts/audit_opportunity_probe.py
$PY scripts/report_opportunity_probe.py --report-dir docs/experiments/EXP-2D-20260926-opportunity-selector-v1 --work-dir outputs/EXP-2D-20260926-opportunity-selector-v1
