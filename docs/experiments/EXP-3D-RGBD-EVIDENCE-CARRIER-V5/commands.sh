#!/usr/bin/env bash
# V5 single-factor dense/RGB-D evidence carrier (EXP-3D-RGBD-EVIDENCE-CARRIER-V5).
# CPU cores 0-7 of this host crashed natively in numpy during V2 pre-freeze checks, so every
# command is pinned to cores 8-23 with single-threaded numeric libraries.
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PIN="taskset -c 8-23"

# 1. Numeric reproduction from the published gzip raw (no model, image or depth is loaded):
$PIN .venv/bin/python scripts/analyze_rgbd_evidence_carrier.py \
  --root docs/experiments/EXP-3D-RGBD-EVIDENCE-CARRIER-V5 --output /tmp/rgbd-evidence-v5-reproduction

# 2. Byte-exact audit of the local scientific JSON against a fresh raw replay:
$PIN .venv/bin/python scripts/audit_rgbd_evidence_statistics.py \
  --root outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5 --output /tmp/rgbd-evidence-v5-audit.json

# 3. The formal run as executed. Rerunning needs the V2-locked TRAIN/DEV data (context depth is
#    carrier input on this RGB-D track), finalized V2-V4 outputs, the V5 protocol text as
#    NEW_OUTPUT/PROTOCOL.md, and a NEW output directory named EXP-3D-RGBD-EVIDENCE-CARRIER-V5
#    under a different parent; never overwrite this sealed run. Create the subdirectories first.
# mkdir -p NEW_PARENT/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/{audit,checkpoints,raw,figures}
# $PIN .venv/bin/python scripts/prepare_rgbd_evidence_experiment.py --root NEW_PARENT/EXP-3D-RGBD-EVIDENCE-CARRIER-V5
# $PIN .venv/bin/python scripts/run_rgbd_evidence_experiment.py --root NEW_PARENT/EXP-3D-RGBD-EVIDENCE-CARRIER-V5 --docs NEW_DOCS/EXP-3D-RGBD-EVIDENCE-CARRIER-V5

# 4. Code checks:
$PIN .venv/bin/python -m pytest -q
