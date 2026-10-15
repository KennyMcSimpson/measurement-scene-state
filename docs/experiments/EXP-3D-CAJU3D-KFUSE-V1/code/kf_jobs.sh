#!/bin/bash
# EXP-3D-CAJU3D-KFUSE-V1 jobs. No confirmation truth is read before the fusion tables are frozen and hashed.
#   root | test | lock      (CPU)  root files | unit tests | preregistration lock
#   rehearsal              (GPU)  H-JEPA query features of 2 FIT_SYSTEM episodes, outside the root
#   features               (GPU)  H-JEPA query features of every cohort (FIT_SYSTEM FIT_SYSTEM2 CONF_SYSTEM3 CONF_VAL)
#   fit                    (CPU)  fusion tables on FIT + leave-one-place-out description
#   confirm                (CPU)  confirmation twice (byte-identical)
# GPU stages run through ~/cv_tools/cv_gpu_run.sh --with-others (GPU 4 only).
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-CAJU3D-KFUSE-V1
DS=$HOME/cv_work/depthsplat/depthsplat
PY=$HOME/envs/depthsplat/bin/python
cpu() {
  (cd $REPO && CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src taskset -c 40-55 .venv/bin/python $HOME/cv_tools/cv_host_run.py "$@")
}
gpu_env() {
  cd $DS || exit 1
  export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
}
case "$1" in
  root) set -e; cpu $E/code/kf_make_root.py $E ;;
  test)
    cd $REPO
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 40-47 .venv/bin/python -m pytest -q -p no:cacheprovider -rA \
      $E/code/test_kf.py > $E/audit/tests.log 2>&1
    rc=$?
    tail -1 $E/audit/tests.log
    exit $rc
    ;;
  lock) set -e; cpu $E/code/kf_lock.py $E ;;
  rehearsal)
    gpu_env
    $PY $E/code/kf_query_jepa.py $E $HOME/cv_work/caju3d/kf_rehearsal --cohorts FIT_SYSTEM --limit 2 \
      > $HOME/cv_work/caju3d/kf_rehearsal.log 2>&1
    echo "KF_REHEARSAL_EXIT $?"
    ;;
  features)
    [ -e $E/preregistration.json ] || { echo "not locked"; exit 1; }
    gpu_env
    fail=0
    for c in FIT_SYSTEM FIT_SYSTEM2 CONF_SYSTEM3 CONF_VAL; do
      pids=()
      for k in 0 1; do
        $PY $E/code/kf_query_jepa.py $E $E/jepa/$c --cohorts $c --shard $k/2 > $E/audit/jepa_${c}_shard$k.log 2>&1 &
        pids+=($!)
      done
      for p in "${pids[@]}"; do wait "$p" || fail=1; done
    done
    echo "KF_FEATURES_EXIT $fail"
    exit $fail
    ;;
  fit) set -e; cpu $E/code/kf_score.py $E fit ;;
  confirm)
    set -e
    cpu $E/code/kf_score.py $E confirm
    cp $E/results/confirm_results.json $E/audit/confirm_results.first_run.json
    cpu $E/code/kf_score.py $E confirm > /dev/null
    cmp $E/results/confirm_results.json $E/audit/confirm_results.first_run.json && echo ANALYSIS_DETERMINISTIC \
      | tee $E/audit/analysis_determinism.txt
    ;;
  *) echo "usage: $0 root|test|lock|rehearsal|features|fit|confirm" >&2; exit 64 ;;
esac
