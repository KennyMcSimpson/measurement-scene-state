#!/bin/bash
# DepthSplat variants on the EVAL34 development probe (EVAL-V3 + EVAL-V4, 83 scenes), same settings as on DEV.
set -u
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD
manifest=~/cv_work/depthsplat/probe_eval34/manifest_eval34.json
ft2=~/cv_work/depthsplat/ft_v2/depthsplat_ft_final.pth
run() {
  local name=$1; shift
  [ -e ~/cv_work/depthsplat/ds_e34_$name ] && { echo "SKIP $name (exists)"; return 0; }
  ~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_predict.py "$manifest" EVAL34 ~/cv_work/depthsplat/ds_e34_$name "$@" \
    > ~/cv_work/depthsplat/ds_e34_$name.log 2>&1 || { echo "FAILED $name"; return 1; }
  grep DS_PREDICTED ~/cv_work/depthsplat/ds_e34_$name.log
}
run rgbd_x9 --extra-rgb 9 --inject-measured-depth &&
run ft2_rgbd_x9 --extra-rgb 9 --inject-measured-depth --checkpoint "$ft2" &&
run rgbd_x9_all --extra-rgb 9 --inject-measured-depth --inject-all &&
run rgbd_x3 --extra-rgb 3 --inject-measured-depth &&
run ft2_rgbd_x3 --extra-rgb 3 --inject-measured-depth --checkpoint "$ft2"
