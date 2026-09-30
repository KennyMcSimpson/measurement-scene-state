#!/usr/bin/env bash
# V4 single-factor dense/plane-sweep carrier (EXP-3D-PLANE-SWEEP-CARRIER-V4).
# CPU cores 0-7 of this host crashed natively in numpy during V2 pre-freeze checks, so every
# command is pinned to cores 8-23 with single-threaded numeric libraries.
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PIN="taskset -c 8-23"

# 1. Numeric reproduction from the published gzip raw (no model, image or depth is loaded):
$PIN .venv/bin/python scripts/analyze_plane_sweep_carrier.py \
  --root docs/experiments/EXP-3D-PLANE-SWEEP-CARRIER-V4 --output /tmp/plane-sweep-v4-reproduction

# 2. Byte-exact audit of the local scientific JSON against a fresh raw replay:
$PIN .venv/bin/python scripts/audit_plane_sweep_statistics.py \
  --root outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4 --output /tmp/plane-sweep-v4-audit.json

# 3. The formal run as executed. Rerunning needs the V2-locked TRAIN/DEV data, the V3 protocol
#    text as NEW_OUTPUT/PROTOCOL.md, and a NEW output directory named
#    EXP-3D-PLANE-SWEEP-CARRIER-V4 under a different parent; never overwrite this sealed run.
#    The prepare script does not create checkpoints/, raw/ or figures/: create them first.
# mkdir -p NEW_PARENT/EXP-3D-PLANE-SWEEP-CARRIER-V4/{audit,checkpoints,raw,figures}
# $PIN .venv/bin/python scripts/prepare_plane_sweep_experiment.py --root NEW_PARENT/EXP-3D-PLANE-SWEEP-CARRIER-V4
# $PIN .venv/bin/python scripts/run_plane_sweep_experiment.py --root NEW_PARENT/EXP-3D-PLANE-SWEEP-CARRIER-V4 --docs NEW_DOCS/EXP-3D-PLANE-SWEEP-CARRIER-V4

# 4. Code checks:
$PIN .venv/bin/python -m pytest -q
