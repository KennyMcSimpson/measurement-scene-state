#!/bin/bash
# EXP-3D-CAJU3D-ENGINE-V1 jobs. No query depth is read before dev-score.
#   root | test | lock                    (CPU)
#   rehearsal                             (GPU) one DEV episode per model into ~/cv_work/caju3d/rehearsal
#   dev                                   (GPU) ZS (CACHE NORM NORM_DEC NORM_DEC_INC) and JN1 (CACHE RAW NORM NORM_DEC
#                                         NORM_DEC_INC) on DEV_SYSTEM + DEV_SYSTEM2, two shards per model
#   dev-score                             (CPU) scoring, selection.json + hash (STOP or the candidate C*)
#   conf                                  (GPU) C* (CACHE + its variant) on CONF_SYSTEM3 + CONF_VAL, four shards
#   conf-score                            (CPU) confirmation twice (byte-identical)
# GPU stages run through ~/cv_tools/cv_gpu_run.sh --with-others (GPU 4 only).
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-CAJU3D-ENGINE-V1
DS=$HOME/cv_work/depthsplat/depthsplat
PY=$HOME/envs/depthsplat/bin/python
cpu() {
  (cd $REPO && CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src taskset -c 40-55 .venv/bin/python $HOME/cv_tools/cv_host_run.py "$@")
}
gpu_env() {
  cd $DS || exit 1
  export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
wait_all() {
  fail=0
  for p in "$@"; do wait "$p" || fail=1; done
  return $fail
}
case "$1" in
  root) set -e; cpu $E/code/engine_make_root.py $E ;;
  test)
    cd $REPO
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 40-47 .venv/bin/python -m pytest -q -p no:cacheprovider -rA \
      $E/code/test_engine.py > $E/audit/tests.log 2>&1
    rc=$?
    tail -1 $E/audit/tests.log
    exit $rc
    ;;
  lock) set -e; cpu $E/code/engine_lock.py $E ;;
  rehearsal)
    gpu_env
    mkdir -p $HOME/cv_work/caju3d
    $PY $E/code/engine_ttt.py $E $HOME/cv_work/caju3d/rehearsal/ZS --model ZS --arms CACHE NORM NORM_DEC NORM_DEC_INC \
      --cohorts DEV_SYSTEM --limit 1 > $HOME/cv_work/caju3d/rehearsal_ZS.log 2>&1 &
    a=$!
    $PY $E/code/engine_ttt.py $E $HOME/cv_work/caju3d/rehearsal/JN1 --model JN1 --arms CACHE RAW NORM NORM_DEC \
      NORM_DEC_INC --cohorts DEV_SYSTEM --limit 1 > $HOME/cv_work/caju3d/rehearsal_JN1.log 2>&1 &
    b=$!
    wait_all $a $b
    echo "ENGINE_REHEARSAL_EXIT $?"
    ;;
  dev)
    gpu_env
    pids=()
    for k in 0 1; do
      $PY $E/code/engine_ttt.py $E $E/engine/DEV/ZS --model ZS --arms CACHE NORM NORM_DEC NORM_DEC_INC \
        --cohorts DEV_SYSTEM DEV_SYSTEM2 --shard $k/2 > $E/audit/dev_ZS_shard$k.log 2>&1 &
      pids+=($!)
      $PY $E/code/engine_ttt.py $E $E/engine/DEV/JN1 --model JN1 --arms CACHE RAW NORM NORM_DEC NORM_DEC_INC \
        --cohorts DEV_SYSTEM DEV_SYSTEM2 --shard $k/2 > $E/audit/dev_JN1_shard$k.log 2>&1 &
      pids+=($!)
    done
    wait_all "${pids[@]}"
    rc=$?
    echo "ENGINE_DEV_GPU_EXIT $rc"
    exit $rc
    ;;
  dev-score) set -e; cpu $E/code/engine_score.py $E dev ;;
  conf)
    sel=$(cd $REPO && .venv/bin/python -c "
import json; s = json.load(open('$E/selection.json'))
assert not s['stop'], 'DEV said STOP'
m, _, v = s['selected'].partition('_'); print(m, v)") || exit 1
    set -- $sel
    gpu_env
    pids=()
    for k in 0 1 2 3; do
      $PY $E/code/engine_ttt.py $E $E/engine/CONF/$1 --model $1 --arms CACHE $2 --cohorts CONF_SYSTEM3 CONF_VAL \
        --shard $k/4 > $E/audit/conf_$1_shard$k.log 2>&1 &
      pids+=($!)
    done
    wait_all "${pids[@]}"
    rc=$?
    echo "ENGINE_CONF_GPU_EXIT $rc"
    exit $rc
    ;;
  conf-score)
    set -e
    cpu $E/code/engine_score.py $E confirm
    cp $E/results/confirm_results.json $E/audit/confirm_results.first_run.json
    cpu $E/code/engine_score.py $E confirm > /dev/null
    cmp $E/results/confirm_results.json $E/audit/confirm_results.first_run.json && echo ANALYSIS_DETERMINISTIC \
      | tee $E/audit/analysis_determinism.txt
    ;;
  *) echo "usage: $0 root|test|lock|rehearsal|dev|dev-score|conf|conf-score" >&2; exit 64 ;;
esac
