#!/bin/bash
# EXP-3D-CAJU3D-JEPAGATE-V1 jobs. No query depth is read before dev-score.
#   root | test | lock      (CPU)  root files copied from ENGINE-V1 | unit tests | preregistration lock
#   rehearsal              (GPU)  2 DEV episodes, arms NONE JEPA DEPTH INC RANDOM, outside the root
#   dev                    (GPU)  DEV_SYSTEM + DEV_SYSTEM2, arms JEPA DEPTH INC RANDOM, 4 shards
#   dev-score              (CPU)  scoring + decision.json (GO / STOP) + hash
#   conf                   (GPU)  CONF_SYSTEM3 + CONF_VAL, arms JEPA DEPTH, 4 shards (only after GO)
#   conf-score             (CPU)  confirmation twice (byte-identical)
# GPU stages run through ~/cv_tools/cv_gpu_run.sh --with-others (GPU 4 only).
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-CAJU3D-JEPAGATE-V1
SRC=$REPO/outputs/EXP-3D-CAJU3D-ENGINE-V1
DS=$HOME/cv_work/depthsplat/depthsplat
PY=$HOME/envs/depthsplat/bin/python
MODEL=$E/model/hjepa_final.pth
cpu() {
  (cd $REPO && CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src taskset -c 40-55 .venv/bin/python $HOME/cv_tools/cv_host_run.py "$@")
}
gpu_env() {
  cd $DS || exit 1
  export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
run_shards() {  # out arms cohorts...
  local out=$1 arms=$2; shift 2
  pids=()
  for k in 0 1 2 3; do
    $PY $E/code/gate_ttt.py $E $out --model $MODEL --arms $arms --cohorts "$@" --shard $k/4 \
      > $E/audit/$(basename $out)_shard$k.log 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  return $fail
}
case "$1" in
  root)
    set -e
    [ ! -e $E/frame_role_lock.json ] || { echo "root files exist"; exit 1; }
    cp $SRC/frame_role_lock.json $SRC/reused_seals.json $E/
    sha256sum $SRC/frame_role_lock.json $SRC/reused_seals.json $E/frame_role_lock.json $E/reused_seals.json \
      > $E/audit/root_copy_sha.txt
    ;;
  test)
    cd $REPO
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 40-47 .venv/bin/python -m pytest -q -p no:cacheprovider -rA \
      $E/code/test_hjepa.py $E/code/test_gates.py $E/code/test_gate_rules.py > $E/audit/tests.log 2>&1
    rc=$?
    tail -1 $E/audit/tests.log
    exit $rc
    ;;
  lock) set -e; cpu $E/code/gate_lock.py $E ;;
  rehearsal)
    gpu_env
    mkdir -p $HOME/cv_work/caju3d
    $PY $E/code/gate_ttt.py $E $HOME/cv_work/caju3d/gate_rehearsal --model $MODEL --arms NONE JEPA DEPTH INC RANDOM \
      --cohorts DEV_SYSTEM --limit 2 > $HOME/cv_work/caju3d/gate_rehearsal.log 2>&1
    echo "GATE_REHEARSAL_EXIT $?"
    ;;
  dev)
    gpu_env
    run_shards $E/gate/DEV "JEPA DEPTH INC RANDOM" DEV_SYSTEM DEV_SYSTEM2
    rc=$?
    echo "GATE_DEV_GPU_EXIT $rc"
    exit $rc
    ;;
  dev-score) set -e; cpu $E/code/gate_score.py $E dev ;;
  conf)
    (cd $REPO && .venv/bin/python -c "
import json, sys; sys.exit(0 if json.load(open('$E/decision.json'))['go'] else 1)") || { echo "DEV said STOP"; exit 1; }
    gpu_env
    run_shards $E/gate/CONF "JEPA DEPTH" CONF_SYSTEM3 CONF_VAL
    rc=$?
    echo "GATE_CONF_GPU_EXIT $rc"
    exit $rc
    ;;
  conf-score)
    set -e
    cpu $E/code/gate_score.py $E confirm
    cp $E/results/confirm_results.json $E/audit/confirm_results.first_run.json
    cpu $E/code/gate_score.py $E confirm > /dev/null
    cmp $E/results/confirm_results.json $E/audit/confirm_results.first_run.json && echo ANALYSIS_DETERMINISTIC \
      | tee $E/audit/analysis_determinism.txt
    ;;
  *) echo "usage: $0 root|test|lock|rehearsal|dev|dev-score|conf|conf-score" >&2; exit 64 ;;
esac
