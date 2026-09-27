#!/usr/bin/env bash
set -euo pipefail
cd /home/zonghan/measurement-scene-state
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/pytest tests/test_mechanism_training.py tests/test_mechanism_calibration.py tests/test_compile_dynamic_training_episodes.py tests/test_pilot_provenance.py tests/test_download.py -q
.venv/bin/ruff check scripts/prepare_3d_training.py scripts/probe_3d_training_resources.py src/mcss/mechanism_pilot/calibration.py src/mcss/mechanism_pilot/training.py tests/test_mechanism_calibration.py tests/test_mechanism_training.py
# Both commands refuse overwrite; choose a fresh suffix for further reproductions.
.venv/bin/python scripts/prepare_3d_training.py --output outputs/EXP-3D-20260927-training-preparation-v1/prepared_reproduction
.venv/bin/python scripts/probe_3d_training_resources.py --output outputs/EXP-3D-20260927-training-preparation-v1/resource_probe_reproduction.json
# No formal training/download command here: scene identities, frame IDs, and adapter integration remain pending.
