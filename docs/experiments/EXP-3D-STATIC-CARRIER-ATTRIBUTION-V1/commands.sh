#!/usr/bin/env bash
# Run from repository root. Default: regenerate statistics from saved raw, no inference.
set -euo pipefail
PY=${PYTHON_BIN:-.venv/bin/python}
EXP=EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1
ORIGINAL=outputs/$EXP
MODE=${1:-summary}
if [[ "$MODE" == summary ]]; then
  "$PY" scripts/report_static_attribution.py --raw-only --output "$ORIGINAL" \
    --docs "${SUMMARY_DIR:-outputs/$EXP-recomputed-summary}"
  exit 0
fi
if [[ "$MODE" != full ]]; then
  echo 'Usage: bash commands.sh [summary|full]' >&2
  exit 2
fi
# Requires original train-data manifest, native files, calibrated metadata and B-final.
# They are preserved locally; see the training-preparation and 1000A+600B reports to restore.
OUT=${REPRO_OUTPUT:-outputs/$EXP-reproduction}
TRAIN=outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training
OLD=outputs/EXP-3D-20260927-small-training-v1/data/manifest.json
CKPT=$TRAIN/phase_b_final.pt
CALIB=outputs/EXP-3D-20260927-training-preparation-v1/metadata/metadata_camera_parameters.csv
PREVIOUS=outputs/EXP-3D-20260927-trained-state-history-v1/run/static_state_results.json
[[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 1; }
mkdir -p "$OUT/metadata" "$OUT/rgb_audit/official_metadata"
cp "$ORIGINAL/preregistration.json" "$ORIGINAL/config.json" "$ORIGINAL/residual_training_amendment.json" "$OUT/"
# Fetch only official pinned metadata; never execute downloaded Python source.
"$PY" - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
import requests
root=Path(sys.argv[1]); commit='3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254'
base=f'https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/{commit}/'
paths={
 'evermotion_dataset/_dataset_config.py':'metadata/upstream_dataset_config.py',
 'README.md':'metadata/upstream_README.md',
 **{f'evermotion_dataset/analysis/{name}':f'rgb_audit/official_metadata/{name}' for name in
 ['metadata_images.csv','metadata_images_split_scene_v1.csv','metadata_camera_trajectories.csv']}}
records=[]
for source,target in paths.items():
 response=requests.get(base+source,timeout=60);response.raise_for_status()
 (root/target).write_bytes(response.content)
 records.append({'url':base+source,'sha256':hashlib.sha256(response.content).hexdigest(),'bytes':len(response.content)})
(root/'rgb_audit/official_metadata/sources.json').write_text(json.dumps(records,indent=2)+'\n')
PY
"$PY" scripts/audit_static_rgb.py --manifest "$OLD" --output "$OUT/rgb_audit" \
  --official-metadata "$OUT/rgb_audit/official_metadata"
"$PY" scripts/audit_static_geometry.py --manifest "$OLD" --checkpoint "$CKPT" \
  --output-dir "$OUT/geometry_audit" --metadata-csv "$CALIB"
# Both fresh runs preserve the pilot and the pre-evaluation budget amendment.
"$PY" scripts/train_static_residual.py --manifest "$OLD" --carrier-checkpoint "$CKPT" \
  --output-dir "$OUT/residual" --steps 3000 --device cuda
"$PY" scripts/train_static_residual.py --manifest "$OLD" --carrier-checkpoint "$CKPT" \
  --output-dir "$OUT/residual_30000" --steps 30000 --device cuda
"$PY" scripts/run_static_attribution.py --manifest "$OLD" --checkpoint "$CKPT" \
  --residual-checkpoint "$OUT/residual_30000/best.pt" --output "$OUT/old_static" --device cuda
"$PY" - "$OUT" "$PREVIOUS" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);old=json.loads(Path(sys.argv[2]).read_text())
new=json.loads((root/'old_static/static_results.json').read_text())
lookup={(r['scene_id'],r['query_id'],r['method']):r for r in new}
metrics=['rgb_mse','rgb_ssim','depth_absrel','depth_rmse','depth_delta1','opacity','coverage']
assert len(old)==90
for r in old:
 n=lookup[r['scene_id'],r['query_id'],r['method']]
 assert all(r[k]==n[k] for k in metrics), 'STOP: old scores mismatch; investigate before unseen'
 assert r['query_camera_hash']==n['query_camera_hash']
(root/'old_static_reproduction.json').write_text(json.dumps({'status':'PASS','rows_compared':90,'exact_numeric_equality':True},indent=2)+'\n')
PY
"$PY" scripts/prepare_static_unseen.py --output "$OUT/unseen_data" --experiment "$OUT" \
  --checkpoint "$CKPT" --training-dir "$TRAIN" --training-manifest "$OLD" \
  --partitions configs/hypersim_er_partitions.csv --calibration "$CALIB"
"$PY" scripts/audit_static_geometry.py --manifest "$OUT/unseen_data/manifest.json" \
  --checkpoint "$CKPT" --output-dir "$OUT/unseen_geometry_audit" --metadata-csv "$CALIB"
"$PY" scripts/run_static_attribution.py --manifest "$OUT/unseen_data/manifest.json" \
  --checkpoint "$CKPT" --residual-checkpoint "$OUT/residual_30000/best.pt" \
  --output "$OUT/unseen_static" --unseen --device cuda
"$PY" scripts/audit_static_unseen_independence.py --experiment "$OUT"
"$PY" scripts/audit_static_opacity.py --root "$OUT" --device cuda
"$PY" scripts/analyze_static_gap.py --manifest "$OLD" --experiment "$OUT" --previous-static-raw "$PREVIOUS"
"$PY" -m pytest -q
"$PY" scripts/report_static_attribution.py --raw-only --output "$OUT" --docs "$OUT/regenerated-summary"
# Full curated report command for the completed experiment (includes test/environment evidence):
# .venv/bin/python scripts/report_static_attribution.py --output outputs/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1 --docs docs/experiments/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1
# No Dynamic TTT command appears here.
