#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
run_dir="$(mktemp -d outputs/trained-state-history-reproduction-XXXXXX)"
.venv/bin/python scripts/run_trained_3d_mechanism.py \
  --manifest outputs/EXP-3D-20260927-small-training-v1/data/manifest.json \
  --checkpoint outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt \
  --output "$run_dir/run" --device cuda
PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest -q
