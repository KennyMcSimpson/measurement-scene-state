#!/bin/bash
# EXP-3D-RGBD-FAR-FOCUS-V26, CPU part: after the GPU training job, the frozen V17 EVAL seal of the four
# cohorts and the score for arm GF (CUDA hidden, AVX2), then the V26 analysis against V21's G1. Meant to
# run under nohup; waits for "FINISHED v26_training" in ~/cv_tools/gpu_wait.log.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror
REPO=$R/measurement-scene-state
V26=$REPO/outputs/EXP-3D-RGBD-FAR-FOCUS-V26
ROOT=$V26/GF/EXP-3D-RGBD-LONG-TRAIN-V17
until grep -q "FINISHED v26_training" ~/cv_tools/gpu_wait.log; do sleep 120; done
grep -q "FINISHED v26_training exit=0" ~/cv_tools/gpu_wait.log || { echo "V26 training failed; no EVAL"; exit 1; }
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 CV_ROOT=$R \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src
core=24
for cohort in EVAL_V3 EVAL_V4 EVAL_FAR FRESH_V2; do
  taskset -c $core .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_long_train_eval.py \
    --root "$ROOT" --stage seal --cohort $cohort > "$V26/audit/seal_GF_$cohort.log" 2>&1 &
  core=$((core + 1))
done
wait
taskset -c 24 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_long_train_eval.py \
  --root "$ROOT" --stage score > "$V26/audit/score_GF.log" 2>&1 || exit 1
taskset -c 24 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/analyze_rgbd_far_focus_v26.py \
  > "$V26/audit/analysis.log" 2>&1
status=$?
echo "V26_EVAL_DONE $(date '+%F %T') exit=$status"
exit $status
