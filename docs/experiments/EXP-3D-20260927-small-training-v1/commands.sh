#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Existing verified data reused; fresh output directory required.
.venv/bin/python scripts/train_small_3d_pilot.py \
  --manifest outputs/EXP-3D-20260927-small-training-v1/data/manifest.json \
  --output-dir outputs/EXP-3D-20260927-small-training-v1/training_reproduction --device cuda
.venv/bin/python scripts/audit_small_3d_training.py \
  --training outputs/EXP-3D-20260927-small-training-v1/training_reproduction \
  --output outputs/EXP-3D-20260927-small-training-v1/audit_reproduction.json
.venv/bin/pytest -q
# Dataset acquisition command is documented by:
.venv/bin/python scripts/prepare_small_3d_pilot.py --help
