#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   gpu:   (through cv_gpu_run.sh) reverse-JEPA training on ARKit TRAIN (three seeds, V1's recipe, concurrently), then
#          V1's locked ds_eval.py on ARKIT160 (CTX): JA1-JA3 (these checkpoints, cap 3.0), J1-J3 (V1, cap 3.0) and
#          ZS3 (released weights, cap 3.0).
#   seal:  (CPU) the arms evaluator seals G1 (three seeds: depth and opacity), REPROJ_HARMONIC and NEAR on ARKIT160.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: jepa_arkit_jobs.sh gpu|seal|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    py=~/envs/depthsplat/bin/python
    mkdir -p $E/checkpoints
    echo "TRAIN_START $(date '+%F %T')"
    declare -A train
    for s in 1 2 3; do
      $py $E/code/train_jepa_arkit.py $E/checkpoints/JA$s --seed $((20261007 + s)) --steps 4000 --lr 5e-6 --warmup 200 \
        --targets 2 --eval-every 500 --scale-max 3.0 > $E/audit/train_JA$s.log 2>&1 &
      train[$s]=$!
    done
    done_seeds=0
    for s in 1 2 3; do
      if wait "${train[$s]}"; then done_seeds=$((done_seeds + 1)); else echo "TRAIN_JA${s}_FAILED"; fi
    done
    echo "TRAIN_END $(date '+%F %T') trained=$done_seeds"
    [ "$done_seeds" -eq 3 ] || { echo "JEPA_ARKIT_GPU_EXIT 1"; exit 1; }
    pids=()
    for s in 1 2 3; do
      $py $V1/code/ds_eval.py $E ARKIT160 $E/ds/ARKIT160/JA$s --model JA$s \
        --checkpoint $E/checkpoints/JA$s/depthsplat_jepa_final.pth --scale-max 3.0 --arms CTX > $E/audit/ds_eval_JA$s.log 2>&1 &
      pids+=($!)
      $py $V1/code/ds_eval.py $E ARKIT160 $E/ds/ARKIT160/J$s --model J$s \
        --checkpoint $V1/checkpoints/J$s/depthsplat_jepa_final.pth --scale-max 3.0 --arms CTX > $E/audit/ds_eval_J$s.log 2>&1 &
      pids+=($!)
    done
    $py $V1/code/ds_eval.py $E ARKIT160 $E/ds/ARKIT160/ZS3 --model ZS3 --scale-max 3.0 --arms CTX \
      > $E/audit/ds_eval_ZS3.log 2>&1 &
    pids+=($!)
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "JEPA_ARKIT_GPU_EXIT $fail"
    exit $fail
    ;;
  seal|score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    if [ "$1" = seal ]; then
      taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $E --stage seal --cohort ARKIT160
    else
      taskset -c 40-55 $RUN $E/code/jepa_arkit_score.py $E score
      taskset -c 40-55 $RUN $E/code/jepa_arkit_score.py $E analyze
      cp $E/results/ARKIT160_results.json $E/audit/ARKIT160_results.first_run.json
      taskset -c 40-55 $RUN $E/code/jepa_arkit_score.py $E analyze > /dev/null
      cmp $E/results/ARKIT160_results.json $E/audit/ARKIT160_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    fi
    ;;
  *) echo "usage: $0 gpu|seal|score" >&2; exit 64 ;;
esac
