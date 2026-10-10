#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   gpu:   (through cv_gpu_run.sh) the JOL1-JOL3 trainings (train_jepa_arkit_overlap_scale.py, JO's settings) concurrently,
#          then V1's locked ds_eval.py (CTX arm, cap 3.0) on ARKIT160_OVERLAP for JOL1-JOL3, concurrently.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: os_jobs.sh gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    py=~/envs/depthsplat/bin/python
    mkdir -p $E/checkpoints
    echo "TRAIN_START $(date '+%F %T')"
    pids=()
    for s in 1 2 3; do
      $py $E/code/train_jepa_arkit_overlap_scale.py $E/checkpoints/JOL$s --seed $((20261007 + s)) --steps 4000 --lr 5e-6 \
        --warmup 200 --targets 2 --eval-every 500 --scale-max 3.0 --hit-table $E/training/hit_shares_SCALE.json \
        --min-hit 0.05 > $E/audit/train_JOL$s.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "TRAIN_END $(date '+%F %T') fail=$fail"
    [ "$fail" -eq 0 ] || { echo "OS_GPU_EXIT 1"; exit 1; }
    pids=()
    for s in 1 2 3; do
      $py $V1/code/ds_eval.py $E/eval_OVERLAP ARKIT160_OVERLAP $E/ds/ARKIT160_OVERLAP/JOL$s --model JOL$s \
        --checkpoint $E/checkpoints/JOL$s/depthsplat_jepa_final.pth --scale-max 3.0 --arms CTX \
        > $E/audit/ds_eval_JOL$s.log 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    grep -h DS_EVAL_DONE $E/audit/ds_eval_JOL*.log
    echo "OS_GPU_EXIT $fail"
    exit $fail
    ;;
  score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    taskset -c 40-55 $RUN $E/code/os_score.py $E score
    taskset -c 40-55 $RUN $E/code/os_score.py $E analyze
    cp $E/results/OS_results.json $E/audit/OS_results.first_run.json
    taskset -c 40-55 $RUN $E/code/os_score.py $E analyze > /dev/null
    cmp $E/results/OS_results.json $E/audit/OS_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    ;;
  *) echo "usage: $0 gpu|score" >&2; exit 64 ;;
esac
