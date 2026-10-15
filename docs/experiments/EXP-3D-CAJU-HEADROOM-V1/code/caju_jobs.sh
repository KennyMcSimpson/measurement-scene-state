#!/bin/bash
# EXP-3D-CAJU-HEADROOM-V1 jobs. No query or history depth is read before a score stage.
#   root         (CPU) caju_make_root.py: cohorts, episodes, contracts (metadata only)
#   geometry     (CPU) caju_geometry.py for every cohort (CAJU_COMPAT replayed against V3's geometry seal)
#   test         (CPU) the unit and contract tests -> tests.json
#   lock         (CPU) caju_lock.py (before any replay and any ground truth)
#   rehearsal    (GPU) one CAJU_PILOT episode into ~/cv_work/caju/rehearsal, outside the root (timing, memory)
#   pilot        (GPU) replay CAJU_PILOT (2 shards) + CAJU_COMPAT, then V1's independent reference runs
#   pilot-score  (CPU) prediction-level equivalence, then the engineering pilot score
#   formal       (GPU) replay CAJU_SEL + CAJU_EVAL in 3 shards
#   score        (CPU) sel -> eval -> analyze twice (byte-identical) -> finalize
# GPU stages run through ~/cv_tools/cv_gpu_run.sh --with-others (GPU 4 only), e.g.
#   ~/cv_tools/cv_gpu_run.sh --with-others --need-gib 20 -- bash caju_jobs.sh formal
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-CAJU-HEADROOM-V1
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
case "$1" in
  root)
    set -e
    cpu $E/code/caju_make_root.py $E
    ;;
  geometry)
    set -e
    for c in CAJU_PILOT CAJU_SEL CAJU_EVAL CAJU_COMPAT; do cpu $E/code/caju_geometry.py $E $c --workers 16; done
    ;;
  test)
    cd $REPO
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 40-47 .venv/bin/python -m pytest -q -p no:cacheprovider -rA \
      $E/code/test_caju.py > $E/audit/tests.log 2>&1
    rc=$?
    .venv/bin/python - "$rc" "$E" <<'EOF'
import json, re, sys, time
rc, root = int(sys.argv[1]), sys.argv[2]
log = open(f"{root}/audit/tests.log").read()
last = [l for l in log.splitlines() if re.search(r"\d+ (passed|failed)", l)][-1]
count = lambda w: int(m.group(1)) if (m := re.search(rf"(\d+) {w}", last)) else 0
json.dump({"command": "pytest -q -p no:cacheprovider -rA code/test_caju.py (repository root, PYTHONPATH=src, CPU)",
           "exit_code": rc, "summary": last.strip(" ="), "passed": count("passed"), "failed": count("failed"),
           "skipped": count("skipped"), "errors": count("error"), "log": "audit/tests.log",
           "time": time.strftime("%Y-%m-%d %H:%M:%S %z")}, open(f"{root}/tests.json", "w"), indent=1)
print("TESTS", last.strip(" ="), "exit", rc)
EOF
    exit $rc
    ;;
  lock)
    set -e
    cpu $E/code/caju_lock.py $E
    ;;
  rehearsal)
    gpu_env
    mkdir -p $HOME/cv_work/caju
    $PY $E/code/caju_replay.py $E $HOME/cv_work/caju/rehearsal --cohorts CAJU_PILOT --limit 1 \
      > $HOME/cv_work/caju/rehearsal.log 2>&1
    echo "CAJU_REHEARSAL_EXIT $?"
    ;;
  pilot)
    gpu_env
    pids=()
    for k in 0 1; do
      $PY $E/code/caju_replay.py $E $E/replay/PILOT --cohorts CAJU_PILOT --shard $k/2 \
        > $E/audit/replay_PILOT_shard$k.log 2>&1 &
      pids+=($!)
    done
    $PY $E/code/caju_replay.py $E $E/replay/COMPAT --cohorts CAJU_COMPAT > $E/audit/replay_COMPAT.log 2>&1 &
    pids+=($!)
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    $PY $E/code/caju_equivalence_gpu.py $E $E/equivalence --episodes 4 > $E/audit/equivalence_gpu.log 2>&1 || fail=1
    echo "CAJU_PILOT_GPU_EXIT $fail"
    exit $fail
    ;;
  pilot-score)
    set -e
    cpu $E/code/caju_equivalence_compare.py $E
    cpu $E/code/caju_score.py $E pilot
    ;;
  formal)
    gpu_env
    pids=()
    for k in 0 1 2; do
      $PY $E/code/caju_replay.py $E $E/replay/FORMAL --cohorts CAJU_SEL CAJU_EVAL --shard $k/3 \
        > $E/audit/replay_FORMAL_shard$k.log 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "CAJU_FORMAL_GPU_EXIT $fail"
    exit $fail
    ;;
  score)
    set -e
    cpu $E/code/caju_score.py $E sel
    cpu $E/code/caju_score.py $E eval
    cpu $E/code/caju_score.py $E analyze
    mkdir -p $E/audit/first_run
    cp $E/results/*.json $E/audit/first_run/
    cpu $E/code/caju_score.py $E analyze > /dev/null
    same=1
    for f in $E/audit/first_run/*.json; do cmp -s "$f" "$E/results/$(basename $f)" || same=0; done
    if [ $same = 1 ]; then echo ANALYSIS_DETERMINISTIC | tee $E/audit/analysis_determinism.txt
    else echo ANALYSIS_NOT_DETERMINISTIC | tee $E/audit/analysis_determinism.txt; fi
    cpu $E/code/caju_finalize.py $E
    ;;
  *)
    echo "usage: $0 root|geometry|test|lock|rehearsal|pilot|pilot-score|formal|score" >&2
    exit 64
    ;;
esac
