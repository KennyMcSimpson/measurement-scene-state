#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1 formal run (after the lock).
#   gpu:   (through cv_gpu_run.sh) NV states of FRESH78 at the DEV-selected c and m: CTX, CACHE, TTT, three shards.
#   score: (CPU) the locked V1 score stage on this root (ZS from V1's seals, NV from this run), then the
#          preregistered analysis twice (byte-identical).
# Usage: ablation_formal.sh gpu|score
set -u
A=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state/outputs/EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1
V1=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
C=$(python3 -c "import json; print(json.load(open('$A/nv_selection.json'))['scale_c'])")
M=$(python3 -c "import json; print(json.load(open('$A/nv_selection.json'))['ttt_mult'])")
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pids=()
    for k in 0 1 2; do
      ~/envs/depthsplat/bin/python $A/code/naive_eval.py $A FRESH78 $A/ds/FRESH78/NV --model NV --scale-c "$C" \
        --opacity 0.95 --arms CTX CACHE TTT --ttt-mult "$M" --ttt-steps 100 --shard $k/3 > $A/audit/nv_eval_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "ABLATION_GPU_EXIT $fail"
    exit $fail
    ;;
  score)
    set -e
    cd /home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    taskset -c 40-55 $RUN $V1/code/score_analyze.py $A score FRESH78 --arms-seal $V1/raw/eval/FRESH78/prediction_seal.json \
      --ds ZS=$V1/ds/FRESH78/ZS NV=$A/ds/FRESH78/NV
    taskset -c 40-55 $RUN $A/code/ablation_analyze.py $A FRESH78 --zs-mult 1 --nv-mult "$M"
    cp $A/results/FRESH78_ablation_results.json $A/audit/FRESH78_ablation_results.first_run.json
    taskset -c 40-55 $RUN $A/code/ablation_analyze.py $A FRESH78 --zs-mult 1 --nv-mult "$M" > /dev/null
    cmp $A/results/FRESH78_ablation_results.json $A/audit/FRESH78_ablation_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    ;;
  *) echo "usage: $0 gpu|score" >&2; exit 64 ;;
esac
