#!/bin/bash
# EXP-MEASUREMENT-PLANNING-ORACLE-V1: the exact commands of the formal run (repository root on hit).
set -euo pipefail
export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror
RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py scripts/run_measurement_planning_oracle_v1.py"
$RUN tests
$RUN lock
for cohort in EVAL34 EVAL34_GAP3 EVAL_FAR REPLICA; do
  taskset -c 40-63 $RUN compute --cohort "$cohort" --workers 24
done
taskset -c 40-63 $RUN analyze
