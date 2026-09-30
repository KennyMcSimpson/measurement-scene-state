#!/usr/bin/env bash
# Commands of the formal run (freeze attempt 2). CPU cores 0-7 of this host crashed natively
# in numpy during pre-freeze checks, so every command is pinned to cores 8-23 with
# single-threaded numeric libraries.
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PIN="taskset -c 8-23"

# 1. Numeric reproduction from the published gzip raw (no model, image or depth is loaded):
$PIN .venv/bin/python scripts/analyze_geometry_carrier.py \
  --root docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2 --output /tmp/geometry-carrier-v2-reproduction

# 2. Byte-exact audit of the local scientific JSON against a fresh raw replay:
$PIN .venv/bin/python scripts/audit_geometry_carrier_statistics.py \
  --root outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2 --output /tmp/geometry-carrier-v2-audit.json

# 3. The formal run exactly as executed. Rerunning needs the locked TRAIN/DEV data and a NEW
#    output directory; never overwrite this sealed run or open protected scenes.
# $PIN .venv/bin/python scripts/prepare_geometry_carrier_experiment.py --root NEW_OUTPUT
# $PIN .venv/bin/python scripts/run_geometry_carrier_experiment.py --root NEW_OUTPUT --docs NEW_DOCS

# 4. Code checks:
$PIN .venv/bin/python -m pytest -q
