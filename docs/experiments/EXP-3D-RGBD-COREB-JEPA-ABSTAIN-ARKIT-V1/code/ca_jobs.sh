#!/bin/bash
# EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1 jobs (after the lock). No query depth is read before `cpu`.
#   gpu: (through cv_gpu_run.sh) V1's locked ds_eval.py, CACHE arm only, cap 0.1, for JN1 (OBJECTIVE-TTT checkpoint) on
#        the DEV cohorts ARKIT_SYSTEM and ARKIT_SYSTEM2, concurrently.
#   cpu: the candidate selection on DEV, then (unless it stops) the confirmation scoring, then the analysis twice
#        (byte-identical).
# Usage: ca_jobs.sh gpu|cpu
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1
O=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
case "$1" in
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pids=()
    for pair in "ARKIT_SYSTEM eval_SYSTEM" "ARKIT_SYSTEM2 eval_SYSTEM2"; do
      set -- $pair
      ~/envs/depthsplat/bin/python $V1/code/ds_eval.py $E/$2 $1 $E/ds/$1/JN1 --model JN1 --scale-max 0.1 --arms CACHE \
        --checkpoint $O/checkpoints/JN1/depthsplat_jepa_final.pth > $E/audit/ds_eval_$1_JN1.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    grep -h DS_EVAL_DONE $E/audit/ds_eval_*.log
    echo "CA_GPU_EXIT $fail"
    exit $fail
    ;;
  cpu)
    set -e
    cd $REPO
    export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
    taskset -c 40-55 $RUN $E/code/ca_score.py $E select
    taskset -c 40-55 $RUN $E/code/ca_score.py $E score
    taskset -c 40-55 $RUN $E/code/ca_score.py $E analyze
    cp $E/results/CA_results.json $E/audit/CA_results.first_run.json
    taskset -c 40-55 $RUN $E/code/ca_score.py $E analyze > /dev/null
    cmp $E/results/CA_results.json $E/audit/CA_results.first_run.json && echo ANALYSIS_DETERMINISTIC
    ;;
  *) echo "usage: $0 gpu|cpu" >&2; exit 64 ;;
esac
