#!/bin/bash
# EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1 jobs. No SYSTEM4 truth is read before every readout is sealed.
#   root      (CPU)  evaluation_contract.json + frame_role_lock.json from the SYSTEM4 manifest (s4_make_root.py)
#   seal      (CPU)  the arms evaluator seal, then V1's geometry_seal.py (REPROJ7_HARMONIC, near7, cosine)
#   gpu       (GPU)  V1's ds_eval.py: released weights, cap 0.1, CACHE + TTT (m = 1, 100 steps), four shards
#   readouts  (CPU)  CORE_B, KF_BASE, FUSE_V1, FUSE_V2 sealed per query (s4_readouts.py)
#   score     (CPU)  the holdout test twice (byte-identical)
# GPU stages run through ~/cv_tools/cv_gpu_run.sh --with-others (GPU 4 only).
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1
V1=$REPO/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
PY=$HOME/envs/depthsplat/bin/python
cd $REPO
export CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror
RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
case "$1" in
  root) set -e; CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 40-55 $RUN $E/code/s4_make_root.py $E ;;
  seal)
    set -e
    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    taskset -c 40-55 $RUN scripts/evaluate_rgbd_carrier_arms.py --root $E --stage seal --cohort ARKIT_SYSTEM4
    taskset -c 40-55 $RUN $V1/code/geometry_seal.py $E ARKIT_SYSTEM4 $E/raw/eval/ARKIT_SYSTEM4/prediction_seal.json \
      --workers 16
    ;;
  gpu)
    cd ~/cv_work/depthsplat/depthsplat || exit 1
    export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pids=()
    for k in 0 1 2 3; do
      $PY $V1/code/ds_eval.py $E ARKIT_SYSTEM4 $E/ds/ARKIT_SYSTEM4/ZS --model ZS --scale-max 0.1 \
        --arms CACHE TTT --ttt-mult 1 --ttt-steps 100 --shard $k/4 > $E/audit/ds_eval_ZS_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "S4_GPU_EXIT $fail"
    exit $fail
    ;;
  readouts) set -e; CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=8 taskset -c 40-55 $PY $E/code/s4_readouts.py $E ;;
  score)
    set -e
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=1 taskset -c 40-55 $PY $E/code/s4_score.py $E
    cp $E/results/holdout_results.json $E/audit/holdout_results.first_run.json
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=1 taskset -c 40-55 $PY $E/code/s4_score.py $E > /dev/null
    cmp $E/results/holdout_results.json $E/audit/holdout_results.first_run.json && echo ANALYSIS_DETERMINISTIC \
      | tee $E/audit/analysis_determinism.txt
    ;;
  *) echo "usage: $0 root|seal|gpu|readouts|score" >&2; exit 64 ;;
esac
