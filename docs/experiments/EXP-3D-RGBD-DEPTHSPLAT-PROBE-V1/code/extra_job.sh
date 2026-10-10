#!/bin/bash
# Extra-frame variants of the DEV DepthSplat probe, in sequence on the assigned GPU (exploratory).
set -u
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD
manifest=~/cv_work/depthsplat/probe_dev/manifest_dev.json
run() {  # name, ds_predict options...
  local name=$1; shift
  ~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_predict.py "$manifest" DEV ~/cv_work/depthsplat/ds_dev_$name "$@" \
    > ~/cv_work/depthsplat/ds_dev_$name.log 2>&1 || { echo "FAILED $name"; return 1; }
  grep DS_PREDICTED ~/cv_work/depthsplat/ds_dev_$name.log
}
run rgb_x3 --extra-rgb 3 &&
run rgbd_x3 --extra-rgb 3 --inject-measured-depth &&
run rgbd_x3_all --extra-rgb 3 --inject-measured-depth --inject-all &&
run rgbd_x9 --extra-rgb 9 --inject-measured-depth &&
run rgbd_x9_all --extra-rgb 9 --inject-measured-depth --inject-all
