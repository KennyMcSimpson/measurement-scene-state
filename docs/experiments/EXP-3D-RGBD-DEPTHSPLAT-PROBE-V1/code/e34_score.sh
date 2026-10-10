#!/bin/bash
# Score the EVAL34 DepthSplat variants once the seal and the GPU runs are done (project env, CPU only).
set -u
d=~/cv_work/depthsplat
until [ -e $d/probe_eval34/raw/eval/EVAL34/prediction_seal.json ] && grep -q E34_DONE $d/e34_job.log; do sleep 30; done
cd ~/zonghan_o_mirror/measurement-scene-state || exit 1
export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror
names=(rgbd_x9 ft2_rgbd_x9 rgbd_x9_all rgbd_x3 ft2_rgbd_x3)
for n in "${names[@]}"; do
  [ -e $d/ds_e34_$n/run_info.json ] || { echo "MISSING $n"; continue; }
  taskset -c 40-47 .venv/bin/python ~/cv_tools/cv_host_run.py $d/ds_score.py $d/probe_eval34 $d/ds_e34_$n > $d/ds_e34_$n/score.out 2>&1
  taskset -c 40-47 .venv/bin/python ~/cv_tools/cv_host_run.py $d/ds_fuse.py $d/probe_eval34 $d/ds_e34_$n 2>/dev/null | tail -8
done
taskset -c 40-47 .venv/bin/python $d/ds_summary.py $d $d/ds_summary_e34.json $(for n in "${names[@]}"; do echo ds_e34_$n; done)
echo E34_SCORED
