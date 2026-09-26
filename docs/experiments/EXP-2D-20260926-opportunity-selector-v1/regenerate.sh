#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONFAULTHANDLER=1
.venv/bin/python scripts/run_opportunity_probe.py report
.venv/bin/python scripts/audit_opportunity_probe.py
.venv/bin/python scripts/report_opportunity_probe.py --report-dir docs/experiments/EXP-2D-20260926-opportunity-selector-v1 --work-dir outputs/EXP-2D-20260926-opportunity-selector-v1
