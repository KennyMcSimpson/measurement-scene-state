#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1 formal GPU job, run through cv_gpu_run.sh after the lock (GPU 4 only).
# Phase 1, concurrently: reverse-JEPA training of the three seeds (TRAIN only); FRESH78 states of the released
#   DepthSplat (ZS: CTX, CACHE and TTT at the DEV-selected step-size multiplier; ZS3: CTX).
# Phase 2: FRESH78 states (CTX, CACHE) of every seed whose training completed (preregistered: a failed seed is
#   reported and the analysis uses the completed seeds when at least two completed).
# No query depth is read by any of these programs.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
C=$R/code
M=$(python3 -c "import json; print(json.load(open('$R/ttt_selection.json'))['selected_mult'])")
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
py=~/envs/depthsplat/bin/python
mkdir -p $R/audit $R/checkpoints
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 60 > $R/audit/gpu_memory.log 2>&1 &
mon=$!
echo "PHASE1_START $(date '+%F %T') ttt_mult=$M"
declare -A train
for s in 1 2 3; do
  $py $C/train_jepa.py $R/checkpoints/J$s --seed $((20261007 + s)) --steps 4000 --lr 5e-6 --warmup 200 --targets 2 \
    --eval-every 500 --scale-max 3.0 > $R/audit/train_J$s.log 2>&1 &
  train[$s]=$!
done
zs=()
for k in 0 1 2; do  # three shards: GPU 4 is time-sliced with a co-tenant process, more contexts get more of it
  $py $C/ds_eval.py $R FRESH78 $R/ds/FRESH78/ZS --model ZS --scale-max 0.1 --arms CTX CACHE TTT --ttt-mult "$M" \
    --ttt-steps 100 --shard $k/3 > $R/audit/ds_eval_ZS_shard$k.log 2>&1 &
  zs+=($!)
done
$py $C/ds_eval.py $R FRESH78 $R/ds/FRESH78/ZS3 --model ZS3 --scale-max 3.0 --arms CTX > $R/audit/ds_eval_ZS3.log 2>&1 &
zs3=$!
fail=0
for p in "${zs[@]}"; do wait "$p" || { echo "ZS_EVAL_FAILED"; fail=1; }; done
wait $zs3 || { echo "ZS3_EVAL_FAILED"; fail=1; }
done_seeds=0
for s in 1 2 3; do
  if wait "${train[$s]}"; then done_seeds=$((done_seeds + 1)); else echo "TRAIN_J${s}_FAILED"; fi
done
echo "PHASE1_END $(date '+%F %T') fail=$fail trained=$done_seeds"
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
echo "PHASE2_END $(date '+%F %T') fail=$fail"
kill $mon
echo "FORMAL_GPU_EXIT $fail"
exit $fail
