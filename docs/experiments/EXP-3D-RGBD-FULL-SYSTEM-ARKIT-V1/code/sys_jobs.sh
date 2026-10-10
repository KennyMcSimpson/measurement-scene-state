#!/bin/bash
# EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   seal:  (CPU) the arms evaluator seals G1 and GA (three seeds each), REPROJ_HARMONIC and NEAR; then V1's locked
#          geometry_seal.py (REPROJ7_HARMONIC, NEAR7, cosine), replayed against the arms seal.
#   gpu:   (through cv_gpu_run.sh) V1's locked ds_eval.py on ARKIT_SYSTEM, released weights (ZS, cap 0.1), CACHE + TTT
#          (m = 1, 100 steps), four shards concurrently.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: sys_jobs.sh seal|gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pids=()
    for k in 0 1 2 3; do
      ~/envs/depthsplat/bin/python $V1/code/ds_eval.py $E ARKIT_SYSTEM $E/ds/ARKIT_SYSTEM/ZS --model ZS --scale-max 0.1 \
        --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/4 > $E/audit/ds_eval_ZS_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "SYS_GPU_EXIT $fail"
    exit $fail
    ;;
  seal|score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    if [ "$1" = seal ]; then
      taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $E --stage seal --cohort ARKIT_SYSTEM
      taskset -c 40-55 $RUN $V1/code/geometry_seal.py $E ARKIT_SYSTEM $E/raw/eval/ARKIT_SYSTEM/prediction_seal.json \
        --workers 16
    else
      taskset -c 40-55 $RUN $E/code/sys_score.py $E score
      taskset -c 40-55 $RUN $E/code/sys_score.py $E analyze
      cp $E/results/ARKIT_SYSTEM_results.json $E/audit/ARKIT_SYSTEM_results.first_run.json
      taskset -c 40-55 $RUN $E/code/sys_score.py $E analyze > /dev/null
      cmp $E/results/ARKIT_SYSTEM_results.json $E/audit/ARKIT_SYSTEM_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    fi
    ;;
  *) echo "usage: $0 seal|gpu|score" >&2; exit 64 ;;
esac
