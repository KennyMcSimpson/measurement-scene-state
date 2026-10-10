#!/bin/bash
# EXP-3D-RGBD-ENSEMBLES-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   seal:     (CPU) the arms evaluator seals G1 (three seeds), REPROJ_HARMONIC and NEAR on ARKIT_VAL.
#   geometry: (CPU) V1's locked geometry_seal.py (REPROJ7_HARMONIC, NEAR7, cosine), replayed against the arms seal.
#   gpu:      (through cv_gpu_run.sh) V1's locked ds_eval.py: the three frozen reverse-JEPA checkpoints (CTX, cap 3.0)
#             and the released DepthSplat (ZS, cap 0.1: CACHE, TTT m = 1, 100 steps, three shards), concurrently.
#   score:    (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: arkit_jobs.sh seal|geometry|gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
K=$REPO/outputs/EXP-3D-RGBD-ENSEMBLES-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pids=()
    for s in 1 2 3; do
      ~/envs/depthsplat/bin/python $V1/code/ds_eval.py $K ARKIT_VAL $K/ds/ARKIT_VAL/J$s --model J$s \
        --checkpoint $V1/checkpoints/J$s/depthsplat_jepa_final.pth --scale-max 3.0 --arms CTX > $K/audit/ds_eval_J$s.log 2>&1 &
      pids+=($!)
    done
    for k in 0 1 2; do
      ~/envs/depthsplat/bin/python $V1/code/ds_eval.py $K ARKIT_VAL $K/ds/ARKIT_VAL/ZS --model ZS --scale-max 0.1 \
        --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/3 > $K/audit/ds_eval_ZS_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "ARKIT_GPU_EXIT $fail"
    exit $fail
    ;;
  seal|geometry|score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    case "$1" in
      seal) taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $K --stage seal --cohort ARKIT_VAL ;;
      geometry) taskset -c 40-55 $RUN $V1/code/geometry_seal.py $K ARKIT_VAL $K/raw/eval/ARKIT_VAL/prediction_seal.json \
                  --workers 16 ;;
      score)
        taskset -c 40-55 $RUN $K/code/arkit_score.py $K score
        taskset -c 40-55 $RUN $K/code/arkit_score.py $K analyze
        cp $K/results/ARKIT_VAL_ensembles_results.json $K/audit/ARKIT_VAL_ensembles_results.first_run.json
        taskset -c 40-55 $RUN $K/code/arkit_score.py $K analyze > /dev/null
        cmp $K/results/ARKIT_VAL_ensembles_results.json $K/audit/ARKIT_VAL_ensembles_results.first_run.json \
          && echo ANALYSIS_DETERMINISTIC ;;
    esac
    ;;
  *) echo "usage: $0 seal|geometry|gpu|score" >&2; exit 64 ;;
esac
