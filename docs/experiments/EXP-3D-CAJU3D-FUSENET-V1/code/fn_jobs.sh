#!/bin/bash
# EXP-3D-CAJU3D-FUSENET-V1 jobs. No confirmation truth is read before every model is trained and its hash logged.
#   root | test | lock      (CPU)  root files copied from KFUSE-V1 | unit tests | preregistration lock
#   rehearsal              (GPU)  H-JEPA query features of 2 FIT_SYSTEM episodes, outside the root
#   features               (GPU)  H-JEPA query features of every cohort
#   train                  (GPU)  BASE and JEPA FuseNets, three seeds each, on FIT
#   confirm                (CPU)  the confirmation twice (byte-identical)
# GPU stages run through ~/cv_tools/cv_gpu_run.sh --with-others (GPU 4 only).
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-CAJU3D-FUSENET-V1
SRC=$REPO/outputs/EXP-3D-CAJU3D-KFUSE-V1
DS=$HOME/cv_work/depthsplat/depthsplat
PY=$HOME/envs/depthsplat/bin/python
cpu() {
  (cd $REPO && CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src taskset -c 40-55 .venv/bin/python $HOME/cv_tools/cv_host_run.py "$@")
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
      $E/code/test_fn.py > $E/audit/tests.log 2>&1
    rc=$?
    tail -1 $E/audit/tests.log
    exit $rc
    ;;
  lock) set -e; cpu $E/code/fn_lock.py $E ;;
  rehearsal)
    cd $DS || exit 1
    PYTHONPATH=$PWD OMP_NUM_THREADS=8 $PY $E/code/fn_query_features.py $E $HOME/cv_work/caju3d/fn_rehearsal \
      --cohorts FIT_SYSTEM --limit 2 > $HOME/cv_work/caju3d/fn_rehearsal.log 2>&1
    echo "FN_REHEARSAL_EXIT $?"
    ;;
  features)
    [ -e $E/preregistration.json ] || { echo "not locked"; exit 1; }
    cd $DS || exit 1
    fail=0
    for c in FIT_SYSTEM FIT_SYSTEM2 CONF_SYSTEM3 CONF_VAL; do
      pids=()
      for k in 0 1; do
        PYTHONPATH=$PWD OMP_NUM_THREADS=8 $PY $E/code/fn_query_features.py $E $E/features/$c --cohorts $c --shard $k/2 \
          > $E/audit/features_${c}_shard$k.log 2>&1 &
        pids+=($!)
      done
      for p in "${pids[@]}"; do wait "$p" || fail=1; done
    done
    echo "FN_FEATURES_EXIT $fail"
    exit $fail
    ;;
  train)
    [ -e $E/preregistration.json ] || { echo "not locked"; exit 1; }
    cd $REPO
    PYTHONPATH=src OMP_NUM_THREADS=8 $PY $E/code/fn_train.py $E > $E/audit/train.log 2>&1
    rc=$?
    tail -8 $E/audit/train.log
    exit $rc
    ;;
  confirm)
    set -e
    cd $REPO
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=8 taskset -c 40-55 $PY $E/code/fn_score.py $E
    cp $E/results/confirm_results.json $E/audit/confirm_results.first_run.json
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=8 taskset -c 40-55 $PY $E/code/fn_score.py $E > /dev/null
    cmp $E/results/confirm_results.json $E/audit/confirm_results.first_run.json && echo ANALYSIS_DETERMINISTIC \
      | tee $E/audit/analysis_determinism.txt
    ;;
  *) echo "usage: $0 root|test|lock|rehearsal|features|train|confirm" >&2; exit 64 ;;
esac
