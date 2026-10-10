#!/bin/bash
# Re-score probe variants with the current ds_score.py (adds FAR_G1 / FAR_DSN_G1), then summarize (CPU only).
set -u
d=~/cv_work/depthsplat
cd ~/zonghan_o_mirror/measurement-scene-state || exit 1
export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror
e34=(rgbd_x9 ft2_rgbd_x9 rgbd_x9_all rgbd_x3 ft2_rgbd_x3)
dev=(256x320 rgbd_256x320 rgb_x3 rgbd_x3 rgbd_x3_all zs_rgb_x9 rgbd_x9 rgbd_x9_all zs_rgbd_x3_align zs_rgbd_x9_align
     ft_base ft_rgb_x3 ft_rgb_x9 ft_rgbd_x3 ft_rgbd_x9 ft2_base ft2_rgb_x3 ft2_rgb_x9 ft2_rgbd_x3 ft2_rgbd_x9 ft2_rgbd_x9_align)
for n in "${e34[@]}"; do
  taskset -c 40-47 .venv/bin/python ~/cv_tools/cv_host_run.py $d/ds_score.py $d/probe_eval34 $d/ds_e34_$n > $d/ds_e34_$n/score.out 2>&1 &
done
for n in "${dev[@]}"; do
  taskset -c 48-51 .venv/bin/python ~/cv_tools/cv_host_run.py $d/ds_score.py $d/probe_dev $d/ds_dev_$n > $d/ds_dev_$n/score.out 2>&1
done
wait
echo "## EVAL34"
.venv/bin/python $d/ds_summary.py $d $d/ds_summary_e34.json $(for n in "${e34[@]}"; do echo ds_e34_$n; done)
echo "## DEV"
.venv/bin/python $d/ds_summary.py $d $d/ds_summary_dev.json $(for n in "${dev[@]}"; do echo ds_dev_$n; done)
echo RESCORED
