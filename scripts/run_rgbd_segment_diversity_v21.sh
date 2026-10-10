#!/bin/bash
# EXP-3D-RGBD-SEGMENT-DIVERSITY-V21, GPU part: six trainings (arms G1 and G5 x three seeds) of the
# frozen V17 L training code on GPU 4, TF32 off. Launched through ~/cv_tools/cv_gpu_run.sh, which picks
# GPU 4 only when it is idle and holds its lock while this runs; the CPU part
# (eval_rgbd_segment_diversity_v21.sh) runs after it.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror
REPO=$R/measurement-scene-state
V21=$REPO/outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21
cd "$REPO" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = "4" ] || { echo "GPU 4 is not held through cv_gpu_run.sh"; exit 1; }
[ -f "$V21/freeze.json" ] || { echo "V21 is not frozen"; exit 1; }
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 CV_ROOT=$R OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src NVIDIA_TF32_OVERRIDE=0
core=56
for arm in G1 G5; do
  for seed in 20260928 20260929 20260930; do
    taskset -c $core .venv/bin/python ~/cv_tools/fp32_run.py scripts/train_rgbd_long_train_carrier.py \
      --root "$V21/$arm/EXP-3D-RGBD-LONG-TRAIN-V17" --variant L --seed $seed --device cuda \
      > "$V21/audit/train_${arm}_$seed.log" 2>&1 &
    core=$((core + 1))
  done
done
wait
status=0
for arm in G1 G5; do
  for seed in 20260928 20260929 20260930; do
    [ -f "$V21/$arm/EXP-3D-RGBD-LONG-TRAIN-V17/checkpoints/L_$seed/summary.json" ] || { echo "MISSING $arm $seed"; status=1; }
  done
done
echo "V21_TRAINING_DONE $(date '+%F %T') status=$status"
exit $status
