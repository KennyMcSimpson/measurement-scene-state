#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1, amendment 1 (2026-10-08): the training and J-evaluation part of
# formal_gpu_job.sh with identical commands and settings. Attempt 1 co-scheduled the three trainings with three ZS
# shards and ZS3; its peak memory exceeded GPU 4's 47.4 GiB and all three trainings died of CUDA OOM at their first
# step (no optimizer step taken). Attempt 1's ZS and ZS3 evaluations were unaffected and are kept. Only the schedule
# changes: this job starts after attempt 1 has released GPU 4, trains the three seeds concurrently, then evaluates
# every completed seed (CTX, CACHE). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True changes allocation only.
# Run through cv_gpu_run.sh. No query depth is read.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
C=$R/code
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=~/envs/depthsplat/bin/python
mkdir -p $R/audit $R/checkpoints
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 60 > $R/audit/gpu_memory_amend1.log 2>&1 &
mon=$!
echo "TRAIN_START $(date '+%F %T')"
declare -A train
for s in 1 2 3; do
  $py $C/train_jepa.py $R/checkpoints/J$s --seed $((20261007 + s)) --steps 4000 --lr 5e-6 --warmup 200 --targets 2 \
    --eval-every 500 --scale-max 3.0 > $R/audit/train_J$s.log 2>&1 &
  train[$s]=$!
done
fail=0
done_seeds=0
for s in 1 2 3; do
  if wait "${train[$s]}"; then done_seeds=$((done_seeds + 1)); else echo "TRAIN_J${s}_FAILED"; fi
done
echo "TRAIN_END $(date '+%F %T') trained=$done_seeds"
[ "$done_seeds" -ge 2 ] || fail=1
pids=()
for s in 1 2 3; do
  ckpt=$R/checkpoints/J$s/depthsplat_jepa_final.pth
  [ -f "$ckpt" ] || continue
  $py $C/ds_eval.py $R FRESH78 $R/ds/FRESH78/J$s --model J$s --checkpoint "$ckpt" --scale-max 3.0 --arms CTX CACHE \
    > $R/audit/ds_eval_J$s.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "J_EVAL_END $(date '+%F %T') fail=$fail"
kill $mon
echo "AMEND1_GPU_EXIT $fail"
exit $fail
