#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   gpu:   (through cv_gpu_run.sh) two pipelines at once, one per model. Each is the accuracy-only reverse-JEPA training
#          (seed 20261008, 4000 steps, scale cap 0.1, last step) followed by V1's locked ds_eval.py on ARKIT_VAL
#          (CACHE + TTT m = 1, 100 steps, cap 0.1), two shards concurrently. JN1: train_jepa_arkit_norm.py; JK1:
#          train_jepa_arkit_cache.py with the precomputed stream table.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: jn_jobs.sh gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
py=~/envs/depthsplat/bin/python
pipeline() {  # NAME SCRIPT [extra training arguments]
  local name=$1 script=$2
  shift 2
  echo "TRAIN_${name}_START $(date '+%F %T')"
  $py $E/code/$script $E/checkpoints/${name}1 --seed 20261008 --steps 4000 --lr 5e-6 --warmup 200 --targets 2 \
    --eval-every 500 --scale-max 0.1 --hit-table $E/training/hit_shares_TRAIN.json --min-hit 0.05 "$@" \
    > $E/audit/train_${name}1.log 2>&1 || { echo "TRAIN_${name}1_FAILED"; return 1; }
  echo "TRAIN_${name}_END $(date '+%F %T')"
  local pids=() fail=0 k p
  for k in 0 1; do
    $py $V1/code/ds_eval.py $E ARKIT_VAL $E/ds/ARKIT_VAL/$name --model $name \
      --checkpoint $E/checkpoints/${name}1/depthsplat_jepa_final.pth --scale-max 0.1 \
      --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/2 > $E/audit/ds_eval_${name}_shard$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "EVAL_${name}_END $(date '+%F %T') fail=$fail"
  return $fail
}
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p $E/checkpoints
    pipeline JN train_jepa_arkit_norm.py > $E/audit/pipeline_JN.log 2>&1 &
    a=$!
    pipeline JK train_jepa_arkit_cache.py --streams $E/training/streams_TRAIN.json > $E/audit/pipeline_JK.log 2>&1 &
    b=$!
    fail=0
    wait $a || fail=1
    wait $b || fail=1
    cat $E/audit/pipeline_JN.log $E/audit/pipeline_JK.log
    echo "JN_GPU_EXIT $fail"
    exit $fail
    ;;
  score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    taskset -c 40-55 $RUN $E/code/jn_score.py $E score
    taskset -c 40-55 $RUN $E/code/jn_score.py $E analyze
    cp $E/results/ARKIT_VAL_results.json $E/audit/ARKIT_VAL_results.first_run.json
    taskset -c 40-55 $RUN $E/code/jn_score.py $E analyze > /dev/null
    cmp $E/results/ARKIT_VAL_results.json $E/audit/ARKIT_VAL_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    ;;
  *) echo "usage: $0 gpu|score" >&2; exit 64 ;;
esac
