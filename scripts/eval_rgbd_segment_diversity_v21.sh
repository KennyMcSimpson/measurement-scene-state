#!/bin/bash
# EXP-3D-RGBD-SEGMENT-DIVERSITY-V21, CPU part: after the GPU training job, the frozen V17 EVAL seal of
# the four cohorts and the score for both arms (CUDA hidden, AVX2), then the V21 analysis. Meant to run
# under nohup; waits for "FINISHED v21_training exit=0" in ~/cv_tools/gpu_wait.log.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror
REPO=$R/measurement-scene-state
V21=$REPO/outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21
until grep -q "FINISHED v21_training" ~/cv_tools/gpu_wait.log; do sleep 120; done
grep -q "FINISHED v21_training exit=0" ~/cv_tools/gpu_wait.log || { echo "V21 training failed; no EVAL"; exit 1; }
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 CV_ROOT=$R \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src
core=24
for arm in G1 G5; do
  for cohort in EVAL_V3 EVAL_V4 EVAL_FAR FRESH_V2; do
    taskset -c $core .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_long_train_eval.py \
      --root "$V21/$arm/EXP-3D-RGBD-LONG-TRAIN-V17" --stage seal --cohort $cohort \
      > "$V21/audit/seal_${arm}_$cohort.log" 2>&1 &
    core=$((core + 1))
  done
done
wait
for arm in G1 G5; do
  taskset -c 24 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_long_train_eval.py \
    --root "$V21/$arm/EXP-3D-RGBD-LONG-TRAIN-V17" --stage score > "$V21/audit/score_$arm.log" 2>&1 || exit 1
done
taskset -c 24 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/analyze_rgbd_segment_diversity_v21.py \
  > "$V21/audit/analysis.log" 2>&1
status=$?
echo "V21_EVAL_DONE $(date '+%F %T') exit=$status"
exit $status
