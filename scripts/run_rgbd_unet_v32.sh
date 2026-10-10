#!/bin/bash
# EXP-3D-RGBD-UNET-CONTEXT-V32, GPU part: three trainings (arm U x three seeds) of the V32
# training script (3D U-Net context branch) on GPU 4, TF32 off. Launched through ~/cv_tools/cv_gpu_run.sh, which picks GPU 4 only
# when it is idle and holds its lock while this runs; the CPU part (eval_rgbd_unet_v32.sh) runs
# after it.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror
REPO=$R/measurement-scene-state
V32=$REPO/outputs/EXP-3D-RGBD-UNET-CONTEXT-V32
ROOT=$V32/U/EXP-3D-RGBD-LONG-TRAIN-V17
cd "$REPO" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = "4" ] || { echo "GPU 4 is not held through cv_gpu_run.sh"; exit 1; }
[ -f "$V32/freeze.json" ] || { echo "V32 is not frozen"; exit 1; }
mkdir -p "$ROOT/checkpoints"
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 CV_ROOT=$R OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src NVIDIA_TF32_OVERRIDE=0
core=56
for seed in 20260928 20260929 20260930; do
  taskset -c $core .venv/bin/python ~/cv_tools/fp32_run.py scripts/train_rgbd_unet_v32.py \
    --root "$ROOT" --variant L --seed $seed --device cuda \
    > "$V32/audit/train_U_$seed.log" 2>&1 &
  core=$((core + 1))
done
wait
status=0
for seed in 20260928 20260929 20260930; do
  [ -f "$ROOT/checkpoints/L_$seed/summary.json" ] || { echo "MISSING U $seed"; status=1; }
done
echo "V32_TRAINING_DONE $(date '+%F %T') status=$status"
exit $status
