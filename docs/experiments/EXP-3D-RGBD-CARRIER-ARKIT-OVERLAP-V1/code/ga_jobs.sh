#!/bin/bash
# EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 jobs (after the freeze). No query depth is read before `score`.
#   train: (CPU, AVX2, one thread each) the frozen V17 L training of GA for three seeds concurrently, cores 56-58.
#   seal:  (CPU) code/ga_make_eval.py, then the arms evaluator seals G1 and GA (three seeds each), REPROJ_HARMONIC and
#          NEAR on ARKIT160_OVERLAP.
#   score: (CPU) scoring after the seal, then the analysis twice (byte-identical).
# Usage: ga_jobs.sh train|seal|score|all
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
C=$REPO/outputs/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1
GA=$C/GA/EXP-3D-RGBD-LONG-TRAIN-V17
cd $REPO || exit 1
export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror PYTHONPATH=src ATEN_CPU_CAPABILITY=avx2 \
  MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
train() {
  echo "GA_TRAIN_START $(date '+%F %T')"
  core=56
  pids=()
  for seed in 20260928 20260929 20260930; do
    taskset -c $core $RUN scripts/train_rgbd_long_train_carrier.py --root $GA --variant L --seed $seed --device cpu \
      > $C/audit/train_GA_$seed.log 2>&1 &
    pids+=($!)
    core=$((core + 1))
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  for seed in 20260928 20260929 20260930; do
    [ -f $GA/checkpoints/L_$seed/summary.json ] || { echo "GA_SEED_MISSING $seed"; fail=1; }
  done
  echo "GA_TRAIN_END $(date '+%F %T') fail=$fail"
  return $fail
}
seal() {
  python3 $C/code/ga_make_eval.py || return 1
  taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $C --stage seal --cohort ARKIT160_OVERLAP
}
score() {
  taskset -c 40-55 $RUN $C/code/ga_score.py $C score || return 1
  taskset -c 40-55 $RUN $C/code/ga_score.py $C analyze || return 1
  cp $C/results/ARKIT160_OVERLAP_carrier_results.json $C/audit/ARKIT160_OVERLAP_carrier_results.first_run.json
  taskset -c 40-55 $RUN $C/code/ga_score.py $C analyze > /dev/null || return 1
  cmp $C/results/ARKIT160_OVERLAP_carrier_results.json $C/audit/ARKIT160_OVERLAP_carrier_results.first_run.json \
    && echo ANALYSIS_DETERMINISTIC
}
case "$1" in
  train) train ;;
  seal) seal ;;
  score) score ;;
  all) train && seal > $C/audit/seal.log 2>&1 && score > $C/audit/score.log 2>&1; echo "GA_ALL_EXIT $?" ;;
  *) echo "usage: $0 train|seal|score|all" >&2; exit 64 ;;
esac
