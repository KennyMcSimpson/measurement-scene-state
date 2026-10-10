#!/bin/bash
# DepthSplat zero-shot predictions for one probe manifest on the GPU cv_gpu_run.sh assigned; logs that GPU's use every 5 s.
# Usage (through cv_gpu_run.sh): gpu_probe.sh MANIFEST SPLIT OUT [ds_predict args...]
set -u
manifest=$1; split=$2; out=$3; shift 3
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD  # DepthSplat imports its own code as the package "src"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > "$out.gpu_util.log" 2>&1 &
monitor=$!
~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_predict.py "$manifest" "$split" "$out" "$@"
status=$?
kill "$monitor"
exit $status
