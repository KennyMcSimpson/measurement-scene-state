#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1 jobs (after the lock). No query depth is read before `score`.
#   gpu:   (through cv_gpu_run.sh) overlap-aware reverse-JEPA training on ARKit TRAIN (three seeds, the JEPA-ARKIT
#          recipe with overlap-filtered targets, concurrently), then V1's locked ds_eval.py on ARKIT160_OVERLAP (CTX,
#          scale cap 3.0) for JO1-JO3 (these checkpoints), JA1-JA3 (JEPA-ARKIT-V1), J1-J3 (V1) and ZS3 (released).
#   seal:  (CPU) the arms evaluator seals G1 (three seeds: depth and opacity), REPROJ_HARMONIC and NEAR.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: overlap_jobs.sh gpu|seal|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1
JA=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1
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
      $py $E/code/train_jepa_arkit_overlap.py $E/checkpoints/JO$s --seed $((20261007 + s)) --steps 4000 --lr 5e-6 \
        --warmup 200 --targets 2 --eval-every 500 --scale-max 3.0 --hit-table $E/training/hit_shares_TRAIN.json \
        --min-hit 0.05 > $E/audit/train_JO$s.log 2>&1 &
      train[$s]=$!
    done
    done_seeds=0
    for s in 1 2 3; do
      if wait "${train[$s]}"; then done_seeds=$((done_seeds + 1)); else echo "TRAIN_JO${s}_FAILED"; fi
    done
    echo "TRAIN_END $(date '+%F %T') trained=$done_seeds"
    [ "$done_seeds" -eq 3 ] || { echo "OVERLAP_GPU_EXIT 1"; exit 1; }
    pids=()
    for s in 1 2 3; do
      $py $V1/code/ds_eval.py $E ARKIT160_OVERLAP $E/ds/ARKIT160_OVERLAP/JO$s --model JO$s \
        --checkpoint $E/checkpoints/JO$s/depthsplat_jepa_final.pth --scale-max 3.0 --arms CTX > $E/audit/ds_eval_JO$s.log 2>&1 &
      pids+=($!)
      $py $V1/code/ds_eval.py $E ARKIT160_OVERLAP $E/ds/ARKIT160_OVERLAP/JA$s --model JA$s \
        --checkpoint $JA/checkpoints/JA$s/depthsplat_jepa_final.pth --scale-max 3.0 --arms CTX > $E/audit/ds_eval_JA$s.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    pids=()
    for s in 1 2 3; do
      $py $V1/code/ds_eval.py $E ARKIT160_OVERLAP $E/ds/ARKIT160_OVERLAP/J$s --model J$s \
        --checkpoint $V1/checkpoints/J$s/depthsplat_jepa_final.pth --scale-max 3.0 --arms CTX > $E/audit/ds_eval_J$s.log 2>&1 &
      pids+=($!)
    done
    $py $V1/code/ds_eval.py $E ARKIT160_OVERLAP $E/ds/ARKIT160_OVERLAP/ZS3 --model ZS3 --scale-max 3.0 --arms CTX \
      > $E/audit/ds_eval_ZS3.log 2>&1 &
    pids+=($!)
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "OVERLAP_GPU_EXIT $fail"
    exit $fail
    ;;
  seal|score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    if [ "$1" = seal ]; then
      taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $E --stage seal --cohort ARKIT160_OVERLAP
    else
      taskset -c 40-55 $RUN $E/code/overlap_score.py $E score
      taskset -c 40-55 $RUN $E/code/overlap_score.py $E analyze
      cp $E/results/ARKIT160_OVERLAP_results.json $E/audit/ARKIT160_OVERLAP_results.first_run.json
      taskset -c 40-55 $RUN $E/code/overlap_score.py $E analyze > /dev/null
      cmp $E/results/ARKIT160_OVERLAP_results.json $E/audit/ARKIT160_OVERLAP_results.first_run.json \
        && echo ANALYSIS_DETERMINISTIC
    fi
    ;;
  *) echo "usage: $0 gpu|seal|score" >&2; exit 64 ;;
esac
