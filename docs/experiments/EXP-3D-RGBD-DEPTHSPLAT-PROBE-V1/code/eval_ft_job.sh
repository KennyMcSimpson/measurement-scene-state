#!/bin/bash
# DEV probe variants with a fine-tuned DepthSplat checkpoint, plus the zero-shot RGB-only x9 row (exploratory).
# Usage (through cv_gpu_run.sh): eval_ft_job.sh TAG CHECKPOINT
set -u
tag=$1; ckpt=$2
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD
manifest=~/cv_work/depthsplat/probe_dev/manifest_dev.json
run() {  # name, ds_predict options...
  local name=$1; shift
  [ -e ~/cv_work/depthsplat/ds_dev_$name ] && { echo "SKIP $name (exists)"; return 0; }
  ~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_predict.py "$manifest" DEV ~/cv_work/depthsplat/ds_dev_$name "$@" \
    > ~/cv_work/depthsplat/ds_dev_$name.log 2>&1 || { echo "FAILED $name"; return 1; }
  grep DS_PREDICTED ~/cv_work/depthsplat/ds_dev_$name.log
}
run zs_rgb_x9 --extra-rgb 9 &&
run ${tag}_base --checkpoint "$ckpt" &&
run ${tag}_rgb_x3 --extra-rgb 3 --checkpoint "$ckpt" &&
run ${tag}_rgb_x9 --extra-rgb 9 --checkpoint "$ckpt" &&
run ${tag}_rgbd_x3 --extra-rgb 3 --inject-measured-depth --checkpoint "$ckpt" &&
run ${tag}_rgbd_x9 --extra-rgb 9 --inject-measured-depth --checkpoint "$ckpt"
