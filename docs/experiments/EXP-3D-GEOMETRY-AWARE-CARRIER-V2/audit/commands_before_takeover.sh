#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export PYTHONPATH=src
# Published numeric reproduction (no model or image/depth load):
.venv/bin/python scripts/analyze_geometry_carrier.py --root docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2 --output /tmp/geometry-carrier-v2-reproduction
# Audit local numeric results:
.venv/bin/python scripts/audit_geometry_carrier_statistics.py --root outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2
# Training commands below require a NEW output directory and a newly reproduced
# locked data/protocol bundle. Do not overwrite sealed original runs.
# .venv/bin/python scripts/prepare_geometry_carrier_experiment.py --root NEW_OUTPUT
# for seed in 20260928 20260929 20260930; do
#   for variant in C0 C1; do
#     .venv/bin/python scripts/train_geometry_carrier.py --root NEW_OUTPUT --variant "$variant" --seed "$seed" --device cuda
#   done
# done
# .venv/bin/python scripts/evaluate_geometry_carrier.py --root NEW_OUTPUT --device cuda
# .venv/bin/python scripts/analyze_geometry_carrier.py --root NEW_OUTPUT
# .venv/bin/python scripts/audit_geometry_carrier_statistics.py --root NEW_OUTPUT
.venv/bin/pytest -q
