#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
run_dir="$(mktemp -d outputs/centered-1000a-600b-reproduction-XXXXXX)"
.venv/bin/python scripts/train_small_3d_pilot.py \
  --manifest outputs/EXP-3D-20260927-small-training-v1/data/manifest.json \
  --config docs/experiments/EXP-3D-20260927-centered-training-1000a-600b-v1/config.json \
  --output-dir "$run_dir/training" --device cuda --spatial-mode anchor-centered
.venv/bin/python scripts/audit_small_3d_training.py \
  --training "$run_dir/training" --output "$run_dir/independent_audit.json" \
  --expected-a 1000 --expected-b 600
