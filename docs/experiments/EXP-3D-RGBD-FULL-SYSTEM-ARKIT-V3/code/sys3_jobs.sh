#!/bin/bash
# EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3 jobs (after the lock). No query depth is read before `score`.
#   seal:  (CPU) the arms evaluator (3-context G1 / GA, REPROJ_HARMONIC, NEAR), then V1's locked geometry_seal.py
#          (REPROJ7_HARMONIC, NEAR7, cosine; replayed against the arms seal), then code/eval_carrier_7view.py (G1 / GA
#          built from context + stream).
#   gpu:   (through cv_gpu_run.sh) V1's locked ds_eval.py, released weights (ZS, cap 0.1), CACHE + TTT (m = 1, 100
#          steps), four shards concurrently.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: sys2_jobs.sh seal|gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pids=()
    for k in 0 1 2 3; do
      ~/envs/depthsplat/bin/python $V1/code/ds_eval.py $E ARKIT_SYSTEM3 $E/ds/ARKIT_SYSTEM3/ZS --model ZS --scale-max 0.1 \
        --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/4 > $E/audit/ds_eval_ZS_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "SYS3_GPU_EXIT $fail"
    exit $fail
    ;;
  seal|score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    if [ "$1" = seal ]; then
      taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $E --stage seal --cohort ARKIT_SYSTEM3
      taskset -c 40-55 $RUN $V1/code/geometry_seal.py $E ARKIT_SYSTEM3 $E/raw/eval/ARKIT_SYSTEM3/prediction_seal.json \
        --workers 16
      PYTHONPATH=src taskset -c 40-47 $RUN $E/code/eval_carrier_7view.py $E ARKIT_SYSTEM3
    else
      taskset -c 40-55 $RUN $E/code/sys3_score.py $E score
      taskset -c 40-55 $RUN $E/code/sys3_score.py $E analyze
      cp $E/results/ARKIT_SYSTEM3_results.json $E/audit/ARKIT_SYSTEM3_results.first_run.json
      taskset -c 40-55 $RUN $E/code/sys3_score.py $E analyze > /dev/null
      cmp $E/results/ARKIT_SYSTEM3_results.json $E/audit/ARKIT_SYSTEM3_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    fi
    ;;
  *) echo "usage: $0 seal|gpu|score" >&2; exit 64 ;;
esac
