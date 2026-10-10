#!/bin/bash
# EVAL34 variants whose extra frames are at least 3 frame indices away from every query frame (exploratory).
set -u
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD
manifest=~/cv_work/depthsplat/probe_eval34/manifest_eval34.json
run() {
  local name=$1; shift
  [ -e ~/cv_work/depthsplat/ds_e34_$name ] && { echo "SKIP $name (exists)"; return 0; }
  ~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_predict.py "$manifest" EVAL34 ~/cv_work/depthsplat/ds_e34_$name "$@" \
    > ~/cv_work/depthsplat/ds_e34_$name.log 2>&1 || { echo "FAILED $name"; return 1; }
  grep DS_PREDICTED ~/cv_work/depthsplat/ds_e34_$name.log
}
run rgbd_x9g3 --extra-rgb 9 --min-query-gap 3 --inject-measured-depth &&
run rgbd_x9g3_all --extra-rgb 9 --min-query-gap 3 --inject-measured-depth --inject-all
