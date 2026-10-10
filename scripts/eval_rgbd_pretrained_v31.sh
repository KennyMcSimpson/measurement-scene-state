#!/bin/bash
# EXP-3D-RGBD-PRETRAINED-FEATURES-V31, CPU part: after the GPU training job, the V31 EVAL seal (the
# frozen V17 EVAL stage with the V31 carrier loader) of the four cohorts and the score for arm P (CUDA
# hidden, AVX2), then the V31 analysis against V21's G1. Meant to run under nohup; waits for
# "FINISHED v31_training" in ~/cv_tools/gpu_wait.log.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror
REPO=$R/measurement-scene-state
V31=$REPO/outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31
ROOT=$V31/P/EXP-3D-RGBD-LONG-TRAIN-V17
until grep -q "FINISHED v31_training" ~/cv_tools/gpu_wait.log; do sleep 120; done
grep -q "FINISHED v31_training exit=0" ~/cv_tools/gpu_wait.log || { echo "V31 training failed; no EVAL"; exit 1; }
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 CV_ROOT=$R \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src
core=40
for cohort in EVAL_V3 EVAL_V4 EVAL_FAR FRESH_V2; do
  taskset -c $core .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_pretrained_eval_v31.py \
    --root "$ROOT" --stage seal --cohort $cohort > "$V31/audit/seal_P_$cohort.log" 2>&1 &
  core=$((core + 1))
done
wait
taskset -c 40 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_pretrained_eval_v31.py \
  --root "$ROOT" --stage score > "$V31/audit/score_P.log" 2>&1 || exit 1
taskset -c 40 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/analyze_rgbd_pretrained_v31.py \
  > "$V31/audit/analysis.log" 2>&1
status=$?
echo "V31_EVAL_DONE $(date '+%F %T') exit=$status"
exit $status
