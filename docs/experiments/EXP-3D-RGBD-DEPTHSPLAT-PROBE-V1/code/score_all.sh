#!/bin/bash
# Score DEV probe output dirs (project env, CPU only) and print the FAR summary lines; args: dir suffixes.
set -u
cd ~/zonghan_o_mirror/measurement-scene-state || exit 1
for n in "$@"; do
  echo "=== $n"
  CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror .venv/bin/python ~/cv_tools/cv_host_run.py \
    ~/cv_work/depthsplat/ds_score.py ~/cv_work/depthsplat/probe_dev ~/cv_work/depthsplat/ds_dev_$n 2>&1 |
    grep -E "scale per|FAR share|^REPROJ_HARMONIC |^DS |^DSN_OR_HARM_SCALED|^FAR_DSN_SCALED|REPROJ_HARMONIC - FAR_DSN_SCALED|REPROJ_HARMONIC - DS \[ALL\]"
done
