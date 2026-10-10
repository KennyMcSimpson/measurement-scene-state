#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1 formal CPU stages (repository root on hit, CUDA hidden, cores 40-55).
#   seal:    the arms evaluator seals G1 (three seeds), REPROJ_HARMONIC and NEAR for FRESH78; then the geometry seal
#            (REPROJ7_HARMONIC, NEAR7, cosine; replayed against the arms seal). No query depth is read.
#   score:   only after the formal GPU job finished: scoring (the first query-depth read), the analysis twice
#            (byte-identical), and the integrity summary.
# Usage: formal_cpu_jobs.sh seal|score
set -euo pipefail
cd /home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
R=outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1
RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
SEAL=$PWD/$R/raw/eval/FRESH78/prediction_seal.json
case "$1" in
  seal)
    taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $PWD/$R --stage seal --cohort FRESH78
    taskset -c 40-55 $RUN $R/code/geometry_seal.py $PWD/$R FRESH78 "$SEAL" --workers 16
    ;;
  score)
    M=$(python3 -c "import json; print(json.load(open('$R/ttt_selection.json'))['selected_mult'])")
    D=$PWD/$R/ds/FRESH78
    J=""
    for s in 1 2 3; do ls $D/J$s/ds_seal_*of*.json > /dev/null 2>&1 && J="$J J$s=$D/J$s"; done
    taskset -c 40-55 $RUN $R/code/score_analyze.py $PWD/$R score FRESH78 --arms-seal "$SEAL" \
      --ds ZS=$D/ZS ZS3=$D/ZS3 $J
    taskset -c 40-55 $RUN $R/code/score_analyze.py $PWD/$R analyze FRESH78 --ttt-mult "$M"
    cp $R/results/FRESH78_results.json $R/audit/FRESH78_results.first_run.json
    taskset -c 40-55 $RUN $R/code/score_analyze.py $PWD/$R analyze FRESH78 --ttt-mult "$M"
    cmp $R/results/FRESH78_results.json $R/audit/FRESH78_results.first_run.json && echo "ANALYSIS_DETERMINISTIC"
    ;;
  *)
    echo "usage: $0 seal|score" >&2; exit 64 ;;
esac
