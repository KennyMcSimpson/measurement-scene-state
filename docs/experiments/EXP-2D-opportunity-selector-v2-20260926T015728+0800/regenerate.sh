#!/usr/bin/env bash
set -euo pipefail
REPORT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$REPORT_DIR/../../.." && pwd)"
WORK_DIR="$REPO_DIR/outputs/$(basename "$REPORT_DIR")"
cd "$REPO_DIR"
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python scripts/report_vision_2d_v2.py --report-dir "$REPORT_DIR" --work-dir "$WORK_DIR"
.venv/bin/python scripts/finalize_vision_2d_v2.py --report-dir "$REPORT_DIR"
