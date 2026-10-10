#!/bin/bash
# EXP-3D-RGBD-PRETRAINED-FEATURES-V31, GPU part: three trainings (arm P x three seeds) of the V31
# training script (frozen DINOv2 features) on GPU 4, TF32 off. Launched through ~/cv_tools/cv_gpu_run.sh, which picks GPU 4 only
# when it is idle and holds its lock while this runs; the CPU part (eval_rgbd_pretrained_v31.sh) runs
# after it.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror
REPO=$R/measurement-scene-state
V31=$REPO/outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31
ROOT=$V31/P/EXP-3D-RGBD-LONG-TRAIN-V17
cd "$REPO" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = "4" ] || { echo "GPU 4 is not held through cv_gpu_run.sh"; exit 1; }
[ -f "$V31/freeze.json" ] || { echo "V31 is not frozen"; exit 1; }
mkdir -p "$ROOT/checkpoints"
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 CV_ROOT=$R OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src NVIDIA_TF32_OVERRIDE=0
core=56
for seed in 20260928 20260929 20260930; do
  taskset -c $core .venv/bin/python ~/cv_tools/fp32_run.py scripts/train_rgbd_pretrained_v31.py \
    --root "$ROOT" --variant L --seed $seed --device cuda \
    > "$V31/audit/train_P_$seed.log" 2>&1 &
  core=$((core + 1))
done
wait
status=0
for seed in 20260928 20260929 20260930; do
  [ -f "$ROOT/checkpoints/L_$seed/summary.json" ] || { echo "MISSING P $seed"; status=1; }
done
echo "V31_TRAINING_DONE $(date '+%F %T') status=$status"
exit $status
