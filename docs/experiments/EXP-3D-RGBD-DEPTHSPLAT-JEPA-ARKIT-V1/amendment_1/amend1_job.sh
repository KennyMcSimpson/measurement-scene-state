#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1, amendment 1: waits for the locked score job, then (CPU) the amendment scoring
# and its analysis twice (byte-identical). Usage: amend1_job.sh
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1
until grep -q JEPA_ARKIT_SCORE_JOB_END $E/audit/score_waiter.log 2>/dev/null; do sleep 30; done
grep -q ANALYSIS_DETERMINISTIC $E/audit/score.log || { echo "AMEND1_SKIPPED: the locked score job did not finish"; exit 1; }
set -e
cd $REPO
export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
taskset -c 40-55 $RUN $E/amendment_1/amend1_score.py $E score
taskset -c 40-55 $RUN $E/amendment_1/amend1_score.py $E analyze
cp $E/amendment_1/ARKIT160_amendment_1_results.json $E/audit/ARKIT160_amendment_1_results.first_run.json
taskset -c 40-55 $RUN $E/amendment_1/amend1_score.py $E analyze > /dev/null
cmp $E/amendment_1/ARKIT160_amendment_1_results.json $E/audit/ARKIT160_amendment_1_results.first_run.json \
  && echo AMEND1_ANALYSIS_DETERMINISTIC
echo AMEND1_JOB_END
