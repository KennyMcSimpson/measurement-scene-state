#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1 jobs (after the lock; CPU only, no model is run): the threshold
# selection on DEV, then the confirmation scoring, then the analysis twice (byte-identical).
# Usage: at_jobs.sh all
set -eu
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1
[ "${1:-}" = all ] || { echo "usage: $0 all" >&2; exit 64; }
[ -f $E/preregistration.json ] || { echo "not locked" >&2; exit 1; }
cd $REPO
export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
taskset -c 40-55 $RUN $E/code/at_score.py $E select
taskset -c 40-55 $RUN $E/code/at_score.py $E score
taskset -c 40-55 $RUN $E/code/at_score.py $E analyze
cp $E/results/AT_results.json $E/audit/AT_results.first_run.json
taskset -c 40-55 $RUN $E/code/at_score.py $E analyze > /dev/null
cmp $E/results/AT_results.json $E/audit/AT_results.first_run.json && echo ANALYSIS_DETERMINISTIC
echo AT_JOBS_END
