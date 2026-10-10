#!/bin/bash
# EXP-3D-RGBD-TTT-ENSEMBLE-V1 jobs (after the lock). No query depth is read before `score`.
#   geometry: (CPU) V1's locked geometry_seal.py on EVAL34 with the Core B streams (REPROJ7_HARMONIC, NEAR7, cosine),
#             replayed against the EVAL34 arms seal of EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1.
#   gpu:      (through cv_gpu_run.sh) the released DepthSplat (ZS, scale cap 0.1) with V1's locked ds_eval.py:
#             CACHE and TTT (m = 1, 100 steps), three shards.
#   score:    (CPU) scoring after both seals, then the analysis twice (byte-identical).
# Usage: tttens_jobs.sh geometry|gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
T=$REPO/outputs/EXP-3D-RGBD-TTT-ENSEMBLE-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
ARMS=$REPO/outputs/EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1/raw/eval/EVAL34/prediction_seal.json
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pids=()
    for k in 0 1 2; do
      ~/envs/depthsplat/bin/python $V1/code/ds_eval.py $T EVAL34 $T/ds/EVAL34/ZS --model ZS --scale-max 0.1 \
        --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/3 > $T/audit/ds_eval_ZS_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "TTT_ENSEMBLE_GPU_EXIT $fail"
    exit $fail
    ;;
  geometry|score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    if [ "$1" = geometry ]; then
      taskset -c 40-55 $RUN $V1/code/geometry_seal.py $T EVAL34 $ARMS --workers 16
    else
      taskset -c 40-55 $RUN $T/code/tttens_score.py $T score
      taskset -c 40-55 $RUN $T/code/tttens_score.py $T analyze
      cp $T/results/EVAL34_ttt_ensemble_results.json $T/audit/EVAL34_ttt_ensemble_results.first_run.json
      taskset -c 40-55 $RUN $T/code/tttens_score.py $T analyze > /dev/null
      cmp $T/results/EVAL34_ttt_ensemble_results.json $T/audit/EVAL34_ttt_ensemble_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    fi
    ;;
  *) echo "usage: $0 geometry|gpu|score" >&2; exit 64 ;;
esac
