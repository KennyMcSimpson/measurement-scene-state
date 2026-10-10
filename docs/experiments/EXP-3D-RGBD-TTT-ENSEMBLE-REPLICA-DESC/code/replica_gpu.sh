#!/bin/bash
# Replica descriptive check: V1 locked ds_eval.py, released DepthSplat (ZS, cap 0.1), CACHE + TTT (m = 1, 100 steps).
set -u
RT=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state/outputs/EXP-3D-RGBD-TTT-ENSEMBLE-REPLICA-DESC
V1=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pids=()
for k in 0 1; do
  ~/envs/depthsplat/bin/python $V1/code/ds_eval.py $RT REPLICA $RT/ds/REPLICA/ZS --model ZS --scale-max 0.1 \
    --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/2 > $RT/audit/ds_eval_ZS_shard$k.log 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "REPLICA_GPU_EXIT $fail"; exit $fail
