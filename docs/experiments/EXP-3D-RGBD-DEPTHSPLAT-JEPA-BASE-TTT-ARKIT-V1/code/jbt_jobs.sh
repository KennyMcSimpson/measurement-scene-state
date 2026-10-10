#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   gpu:   (through cv_gpu_run.sh) V1's locked ds_eval.py on ARKIT_VAL, CACHE + TTT (m = 1, 100 steps), scale cap 3.0:
#          JA (JA1 of EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1) and ZS3 (released weights), three shards each, concurrently.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: jbt_jobs.sh gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
B=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
JA=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1/checkpoints/JA1/depthsplat_jepa_final.pth
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    py=~/envs/depthsplat/bin/python
    pids=()
    for k in 0 1 2; do
      $py $V1/code/ds_eval.py $B ARKIT_VAL $B/ds/ARKIT_VAL/JA --model JA --checkpoint $JA --scale-max 3.0 \
        --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/3 > $B/audit/ds_eval_JA_shard$k.log 2>&1 &
      pids+=($!)
      $py $V1/code/ds_eval.py $B ARKIT_VAL $B/ds/ARKIT_VAL/ZS3 --model ZS3 --scale-max 3.0 \
        --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/3 > $B/audit/ds_eval_ZS3_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "JBT_GPU_EXIT $fail"
    exit $fail
    ;;
  score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    taskset -c 40-55 $RUN $B/code/jbt_score.py $B score
    taskset -c 40-55 $RUN $B/code/jbt_score.py $B analyze
    cp $B/results/ARKIT_VAL_results.json $B/audit/ARKIT_VAL_results.first_run.json
    taskset -c 40-55 $RUN $B/code/jbt_score.py $B analyze > /dev/null
    cmp $B/results/ARKIT_VAL_results.json $B/audit/ARKIT_VAL_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    ;;
  *) echo "usage: $0 gpu|score" >&2; exit 64 ;;
esac
