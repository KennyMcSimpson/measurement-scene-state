#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1 jobs (after the lock). No query depth is read before `score`.
#   gpu:   (through cv_gpu_run.sh) at once: three pipelines, each the training of one JL seed (train_jepa_arkit_scale.py,
#          JN's settings, hit table training/hit_shares_SCALE.json) followed by V1's locked ds_eval.py (CACHE arm only,
#          cap 0.1) on ARKIT_SYSTEM3 and on ARKIT_VAL; and the ARKIT_VAL CACHE evaluation of NF's JN2 and JN3.
#   score: (CPU) scoring after every seal, then the analysis twice (byte-identical).
# Usage: ns_jobs.sh gpu|score
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1
NF=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
py=~/envs/depthsplat/bin/python
cache_eval() {  # COHORT SUB-ROOT NAME CHECKPOINT
  local cohort=$1 sub=$2 name=$3 checkpoint=$4
  $py $V1/code/ds_eval.py $E/$sub $cohort $E/ds/$cohort/$name --model "$name" --scale-max 0.1 --arms CACHE \
    --checkpoint "$checkpoint" > $E/audit/ds_eval_${cohort}_$name.log 2>&1
}
train_then_eval() {  # NAME SEED
  local name=$1 seed=$2 checkpoint=$E/checkpoints/$1/depthsplat_jepa_final.pth
  echo "TRAIN_${name}_START $(date '+%F %T')"
  $py $E/code/train_jepa_arkit_scale.py $E/checkpoints/$name --seed "$seed" --steps 4000 --lr 5e-6 --warmup 200 \
    --targets 2 --eval-every 500 --scale-max 0.1 --hit-table $E/training/hit_shares_SCALE.json --min-hit 0.05 \
    > $E/audit/train_$name.log 2>&1 || { echo "TRAIN_${name}_FAILED"; return 1; }
  echo "TRAIN_${name}_END $(date '+%F %T')"
  cache_eval ARKIT_SYSTEM3 eval_SYSTEM3 "$name" "$checkpoint" || { echo "EVAL_SYSTEM3_${name}_FAILED"; return 1; }
  cache_eval ARKIT_VAL eval_VAL "$name" "$checkpoint" || { echo "EVAL_VAL_${name}_FAILED"; return 1; }
  echo "EVAL_${name}_END $(date '+%F %T')"
}
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p $E/checkpoints
    pids=()
    (cache_eval ARKIT_VAL eval_VAL JN2 $NF/checkpoints/JN2/depthsplat_jepa_final.pth \
      && cache_eval ARKIT_VAL eval_VAL JN3 $NF/checkpoints/JN3/depthsplat_jepa_final.pth) &
    pids+=($!)
    for pair in "JL1 20261008" "JL2 20261009" "JL3 20261010"; do
      set -- $pair
      train_then_eval "$1" "$2" > $E/audit/pipeline_$1.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    cat $E/audit/pipeline_JL*.log
    grep -h DS_EVAL_DONE $E/audit/ds_eval_*.log
    echo "NS_GPU_EXIT $fail"
    exit $fail
    ;;
  score)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    taskset -c 40-55 $RUN $E/code/ns_score.py $E score
    taskset -c 40-55 $RUN $E/code/ns_score.py $E analyze
    cp $E/results/NS_results.json $E/audit/NS_results.first_run.json
    taskset -c 40-55 $RUN $E/code/ns_score.py $E analyze > /dev/null
    cmp $E/results/NS_results.json $E/audit/NS_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    ;;
  *) echo "usage: $0 gpu|score" >&2; exit 64 ;;
esac
