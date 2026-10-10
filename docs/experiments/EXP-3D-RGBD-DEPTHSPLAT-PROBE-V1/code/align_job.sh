#!/bin/bash
set -u
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD
manifest=~/cv_work/depthsplat/probe_dev/manifest_dev.json
ft2=~/cv_work/depthsplat/ft_v2/depthsplat_ft_final.pth
run() {
  local name=$1; shift
  [ -e ~/cv_work/depthsplat/ds_dev_$name ] && { echo "SKIP $name (exists)"; return 0; }
  ~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_predict.py "$manifest" DEV ~/cv_work/depthsplat/ds_dev_$name "$@" \
    > ~/cv_work/depthsplat/ds_dev_$name.log 2>&1 || { echo "FAILED $name"; return 1; }
  grep DS_PREDICTED ~/cv_work/depthsplat/ds_dev_$name.log
}
run zs_rgbd_x3_align --extra-rgb 3 --inject-measured-depth --align-extra-scale &&
run zs_rgbd_x9_align --extra-rgb 9 --inject-measured-depth --align-extra-scale &&
run ft2_rgbd_x9_align --extra-rgb 9 --inject-measured-depth --align-extra-scale --checkpoint "$ft2"
