#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   gpu:   (through cv_gpu_run.sh) at once: V1's locked ds_eval.py, CACHE arm only, cap 0.1, on ARKIT_SYSTEM3 for ZS
#          (released weights), JN1 and JK1 (OBJECTIVE-TTT checkpoints); and two pipelines, each the training of one new
#          seed of the JN recipe (train_jepa_arkit_norm.py, JN1's settings) followed by its CACHE evaluation: JN2
#          (seed 20261009) and JN3 (seed 20261010).
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: nf_jobs.sh gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1
O=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
py=~/envs/depthsplat/bin/python
cache_eval() {  # NAME CHECKPOINT ("" = the released weights)
  local name=$1 checkpoint=$2
  local args=(--model "$name" --scale-max 0.1 --arms CACHE)
  [ -n "$checkpoint" ] && args+=(--checkpoint "$checkpoint")
  $py $V1/code/ds_eval.py $E ARKIT_SYSTEM3 $E/ds/ARKIT_SYSTEM3/$name "${args[@]}" > $E/audit/ds_eval_$name.log 2>&1
}
train_then_eval() {  # NAME SEED
  local name=$1 seed=$2
  echo "TRAIN_${name}_START $(date '+%F %T')"
  $py $E/code/train_jepa_arkit_norm.py $E/checkpoints/$name --seed "$seed" --steps 4000 --lr 5e-6 --warmup 200 \
    --targets 2 --eval-every 500 --scale-max 0.1 --hit-table $E/training/hit_shares_TRAIN.json --min-hit 0.05 \
    > $E/audit/train_$name.log 2>&1 || { echo "TRAIN_${name}_FAILED"; return 1; }
  echo "TRAIN_${name}_END $(date '+%F %T')"
  cache_eval "$name" $E/checkpoints/$name/depthsplat_jepa_final.pth
}
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p $E/checkpoints
    pids=()
    cache_eval ZS "" &
    pids+=($!)
    cache_eval JN1 $O/checkpoints/JN1/depthsplat_jepa_final.pth &
    pids+=($!)
    cache_eval JK1 $O/checkpoints/JK1/depthsplat_jepa_final.pth &
    pids+=($!)
    train_then_eval JN2 20261009 > $E/audit/pipeline_JN2.log 2>&1 &
    pids+=($!)
    train_then_eval JN3 20261010 > $E/audit/pipeline_JN3.log 2>&1 &
    pids+=($!)
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    cat $E/audit/pipeline_JN2.log $E/audit/pipeline_JN3.log
    grep -h DS_EVAL_DONE $E/audit/ds_eval_*.log
    echo "NF_GPU_EXIT $fail"
    exit $fail
    ;;
  score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    taskset -c 40-55 $RUN $E/code/nf_score.py $E score
    taskset -c 40-55 $RUN $E/code/nf_score.py $E analyze
    cp $E/results/ARKIT_SYSTEM3_results.json $E/audit/ARKIT_SYSTEM3_results.first_run.json
    taskset -c 40-55 $RUN $E/code/nf_score.py $E analyze > /dev/null
    cmp $E/results/ARKIT_SYSTEM3_results.json $E/audit/ARKIT_SYSTEM3_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    ;;
  *) echo "usage: $0 gpu|score" >&2; exit 64 ;;
esac
