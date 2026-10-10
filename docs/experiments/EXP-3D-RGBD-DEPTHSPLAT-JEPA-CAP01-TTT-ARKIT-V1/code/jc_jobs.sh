#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   gpu:   (through cv_gpu_run.sh) the reverse-JEPA training of JC1 (OVERLAP-V1's train_jepa_arkit_overlap.py, seed
#          20261008, scale cap 0.1), then V1's locked ds_eval.py on ARKIT_VAL (CACHE + TTT m = 1, 100 steps, cap 0.1)
#          for JC1, three shards concurrently.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: jc_jobs.sh gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    py=~/envs/depthsplat/bin/python
    mkdir -p $E/checkpoints
    echo "TRAIN_START $(date '+%F %T')"
    $py $E/code/train_jepa_arkit_overlap.py $E/checkpoints/JC1 --seed 20261008 --steps 4000 --lr 5e-6 --warmup 200 \
      --targets 2 --eval-every 500 --scale-max 0.1 --hit-table $E/training/hit_shares_TRAIN.json --min-hit 0.05 \
      > $E/audit/train_JC1.log 2>&1 || { echo "TRAIN_JC1_FAILED"; echo "JC_GPU_EXIT 1"; exit 1; }
    echo "TRAIN_END $(date '+%F %T')"
    pids=()
    for k in 0 1 2; do
      $py $V1/code/ds_eval.py $E ARKIT_VAL $E/ds/ARKIT_VAL/JC --model JC \
        --checkpoint $E/checkpoints/JC1/depthsplat_jepa_final.pth --scale-max 0.1 \
        --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/3 > $E/audit/ds_eval_JC_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "JC_GPU_EXIT $fail"
    exit $fail
    ;;
  score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    taskset -c 40-55 $RUN $E/code/jc_score.py $E score
    taskset -c 40-55 $RUN $E/code/jc_score.py $E analyze
    cp $E/results/ARKIT_VAL_results.json $E/audit/ARKIT_VAL_results.first_run.json
    taskset -c 40-55 $RUN $E/code/jc_score.py $E analyze > /dev/null
    cmp $E/results/ARKIT_VAL_results.json $E/audit/ARKIT_VAL_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    ;;
  *) echo "usage: $0 gpu|score" >&2; exit 64 ;;
esac
