#!/bin/bash
# DepthSplat depth fine-tuning on the GPU cv_gpu_run.sh assigned; logs that GPU's use every 10 s.
# Usage (through cv_gpu_run.sh): ft_job.sh OUT [ds_finetune args...]
set -u
out=$1; shift
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 10 > "$out.gpu_util.log" 2>&1 &
monitor=$!
~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_finetune.py "$out" "$@"
status=$?
kill "$monitor"
exit $status
