#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
# Use a fresh output directory; training refuses to overwrite any existing run.
run_dir="$(mktemp -d outputs/centered-support-reproduction-XXXXXX)"
.venv/bin/python scripts/diagnose_3d_observed_support.py \
  --manifest outputs/EXP-3D-20260927-small-training-v1/data/manifest.json \
  --output "$run_dir/observed_support.json"
.venv/bin/python scripts/train_small_3d_pilot.py \
  --manifest outputs/EXP-3D-20260927-small-training-v1/data/manifest.json \
  --output-dir "$run_dir/training" --device cuda --spatial-mode anchor-centered
.venv/bin/python scripts/audit_small_3d_training.py \
  --training "$run_dir/training" --output "$run_dir/independent_audit.json"
.venv/bin/pytest -q
