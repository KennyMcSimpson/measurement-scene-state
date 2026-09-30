#!/usr/bin/env bash
# Run from repo root. Never opens final holdout. Never trains or runs Dynamic TTT.
set -euo pipefail
PY=${PYTHON_BIN:-.venv/bin/python}
EXP=EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1
ROOT=outputs/$EXP
PREVIOUS=outputs/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1
MODE=${1:-summary}
case "$MODE" in
  summary)
    "$PY" scripts/report_support_redesign.py --raw-only --output "$ROOT" \
      --docs "${SUMMARY_DIR:-outputs/$EXP-recomputed-summary}"
    ;;
  reproduce)
    # Repeats the already-exposed DEVELOPMENT experiment under its immutable lock.
    # The inference runner independently rejects every final-holdout scene identity.
    OUT=${REPRO_OUTPUT:-outputs/$EXP-oracle-reproduction}
    [[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 1; }
    "$PY" scripts/run_support_redesign_oracles.py --experiment "$ROOT" \
      --dev-manifest "$ROOT/data/dev_manifest.json" \
      --exposed-manifest "$PREVIOUS/unseen_data/manifest.json" \
      --checkpoint outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt \
      --output "$OUT" --device cuda
    "$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
from mcss.mechanism_pilot.support_redesign_statistics import analyze
p=Path(sys.argv[1]);raw=json.loads((p/'results.json').read_text())
(p/'bootstrap_results.json').write_text(json.dumps(analyze(raw),indent=2)+'\n')
print('DEV statistics generated; final holdout was not opened')
PY
    ;;
  tests)
    "$PY" -m pytest -q
    ;;
  *)
    echo 'Usage: bash commands.sh [summary|reproduce|tests]' >&2
    exit 2
    ;;
esac

# Audit and plot the original completed run (derived reports only, no model forward):
# .venv/bin/python scripts/audit_support_redesign.py --root outputs/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1
# .venv/bin/python scripts/plot_support_redesign.py --root outputs/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1
# .venv/bin/python scripts/report_support_redesign.py --output outputs/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1 --docs docs/experiments/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1

# If restoring data on a different machine: create a FRESH experiment directory,
# copy preregistration.json there, and run the following with --output NEW_ROOT/data.
# --experiment is the PREVIOUS static experiment holding pinned upstream metadata,
# not the new output root. The preparation refuses overwrite and locks all candidates.
# It reads only official metadata, camera/GT validity and geometry; never model scores.
# .venv/bin/python scripts/prepare_support_redesign_data.py \
#   --output NEW_ROOT/data \
#   --experiment outputs/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1 \
#   --checkpoint outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt \
#   --training-dir outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training \
#   --training-manifest outputs/EXP-3D-20260927-small-training-v1/data/manifest.json \
#   --partitions configs/hypersim_er_partitions.csv \
#   --calibration outputs/EXP-3D-20260927-training-preparation-v1/metadata/metadata_camera_parameters.csv
# The historical exposure audit is a separate requirement. Successful preparation
# does NOT permit final-holdout inference or certify project-level independence.
