#!/usr/bin/env bash
set -euo pipefail
REPORT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$REPORT_DIR/../../.." && pwd)"
cd "$REPO_DIR"
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
case "${1:-analysis}" in
  analysis) bash "$REPORT_DIR/regenerate.sh" ;;
  audit) .venv/bin/python scripts/audit_vision_2d_v2.py --paths "$REPORT_DIR/paths.json" ;;
  tests) .venv/bin/pytest -q ;;
  discovery)
    # Recompute only authorized discovery, in new directories; never overwrite this run.
    REPLAY_ID="EXP-2D-opportunity-selector-v2-replay-$(date +%Y%m%dT%H%M%S%z)"
    REPLAY_REPORT="$REPO_DIR/docs/experiments/$REPLAY_ID"
    REPLAY_WORK="$REPO_DIR/outputs/$REPLAY_ID"
    mkdir "$REPLAY_REPORT" "$REPLAY_WORK"
    cp "$REPORT_DIR/config.json" "$REPORT_DIR/preregistration.json" "$REPORT_DIR/feature_contract.json" "$REPORT_DIR/independent_data_audit.json" "$REPORT_DIR/predictions_manifest.json" "$REPLAY_REPORT/"
    .venv/bin/python - "$REPLAY_REPORT" "$REPLAY_WORK" <<'PYCODE'
import json,sys
from pathlib import Path
report,work=sys.argv[1:]
Path(report,'paths.json').write_text(json.dumps({'report':report,'work':work},indent=2)+'\n')
PYCODE
    .venv/bin/python scripts/run_vision_2d_v2.py discovery --paths "$REPLAY_REPORT/paths.json"
    .venv/bin/python scripts/run_vision_2d_v2.py fit --paths "$REPLAY_REPORT/paths.json"
    .venv/bin/python scripts/report_vision_2d_v2.py --report-dir "$REPLAY_REPORT" --work-dir "$REPLAY_WORK"
    ;;
  *) echo "Use analysis|audit|tests|discovery" >&2; exit 2 ;;
esac
# Pending confirmation interfaces (NOT runnable with the current blocked manifest):
# .venv/bin/python scripts/run_vision_2d_v2.py predict --paths "$REPORT_DIR/paths.json" --manifest QUALIFIED_PRE_GT_LOCKED_MANIFEST.json
# .venv/bin/python scripts/run_vision_2d_v2.py evaluate --paths "$REPORT_DIR/paths.json" --manifest QUALIFIED_PRE_GT_LOCKED_MANIFEST.json
# Before these: qualify original-video provenance, exact/near duplicates, license, fixed
# mask decoder/remote fetch adapter; freeze all implementation+selector+frame artifacts.
# Current v2_dataset.SUPPORTS_REMOTE_TARGET_FETCH=False deliberately refuses evaluation.
# No instruction here permits opening reserve/DAVIS official validation.
