#!/bin/bash
# EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1 jobs (after the lock).
#   gpu:   (through cv_gpu_run.sh) the three frozen V1 reverse-JEPA checkpoints, CTX arm, on EVAL34, with V1's locked
#          ds_eval.py; no query depth is read.
#   seal:  (CPU) the arms evaluator seals G1 (three seeds), REPROJ_HARMONIC and NEAR on EVAL34.
#   score: (CPU) scoring after both seals, then the analysis twice (byte-identical).
# Usage: ens_jobs.sh gpu|seal|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pids=()
    for s in 1 2 3; do
      ~/envs/depthsplat/bin/python $V1/code/ds_eval.py $E EVAL34 $E/ds/EVAL34/J$s --model J$s \
        --checkpoint $V1/checkpoints/J$s/depthsplat_jepa_final.pth --scale-max 3.0 --arms CTX > $E/audit/ds_eval_J$s.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "ENSEMBLE_GPU_EXIT $fail"
    exit $fail
    ;;
  seal|score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    if [ "$1" = seal ]; then
      taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $E --stage seal --cohort EVAL34
    else
      taskset -c 40-55 $RUN $E/code/ens_score.py $E score
      taskset -c 40-55 $RUN $E/code/ens_score.py $E analyze
      cp $E/results/EVAL34_ensemble_results.json $E/audit/EVAL34_ensemble_results.first_run.json
      taskset -c 40-55 $RUN $E/code/ens_score.py $E analyze > /dev/null
      cmp $E/results/EVAL34_ensemble_results.json $E/audit/EVAL34_ensemble_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    fi
    ;;
  *) echo "usage: $0 gpu|seal|score" >&2; exit 64 ;;
esac
