#!/bin/bash
# EXP-3D-RGBD-UNET-CONTEXT-V32, CPU part: after the GPU training job, the V32 EVAL seal (the frozen
# V17 EVAL stage with the V32 carrier loader) of the four cohorts and the score for arm U (CUDA hidden,
# AVX2), then the V32 analysis against V21's G1. Meant to run under nohup; waits for
# "FINISHED v32_training" in ~/cv_tools/gpu_wait.log.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror
REPO=$R/measurement-scene-state
V32=$REPO/outputs/EXP-3D-RGBD-UNET-CONTEXT-V32
ROOT=$V32/U/EXP-3D-RGBD-LONG-TRAIN-V17
until grep -q "FINISHED v32_training" ~/cv_tools/gpu_wait.log; do sleep 120; done
grep -q "FINISHED v32_training exit=0" ~/cv_tools/gpu_wait.log || { echo "V32 training failed; no EVAL"; exit 1; }
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 CV_ROOT=$R \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src
core=44
for cohort in EVAL_V3 EVAL_V4 EVAL_FAR FRESH_V2; do
  taskset -c $core .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_unet_eval_v32.py \
    --root "$ROOT" --stage seal --cohort $cohort > "$V32/audit/seal_U_$cohort.log" 2>&1 &
  core=$((core + 1))
done
wait
taskset -c 44 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_unet_eval_v32.py \
  --root "$ROOT" --stage score > "$V32/audit/score_U.log" 2>&1 || exit 1
taskset -c 44 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/analyze_rgbd_unet_v32.py \
  > "$V32/audit/analysis.log" 2>&1
status=$?
echo "V32_EVAL_DONE $(date '+%F %T') exit=$status"
exit $status
