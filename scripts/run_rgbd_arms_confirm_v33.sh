#!/bin/bash
# EXP-3D-RGBD-ARMS-CONFIRM-V33, CPU only: waits for V32's evaluation, checks the preregistered run
# condition (V32 P1 mean > 0) and the V33 lock; if both hold, seals the three cohorts in parallel
# (CUDA hidden, AVX2, cores 48-50), scores and analyzes; otherwise records that V33 was not run.
set -u
R=/home/xmum26_zonghan/zonghan_o_mirror
REPO=$R/measurement-scene-state
V33=$REPO/outputs/EXP-3D-RGBD-ARMS-CONFIRM-V33
V32_RESULTS=$REPO/outputs/EXP-3D-RGBD-UNET-CONTEXT-V32/results.json
cd "$REPO" || exit 1
until grep -q "V32_EVAL_DONE" ~/cv_work/v32/eval_job.out 2>/dev/null; do sleep 120; done
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 CV_ROOT=$R \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src
.venv/bin/python - "$V33" "$V32_RESULTS" <<'EOF'
import hashlib, importlib.util, json, sys
from pathlib import Path
root, v32 = Path(sys.argv[1]), Path(sys.argv[2])
lock = json.loads((root / "preregistration.json").read_text())
for name, digest in lock["source_sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Frozen source changed: {name}")
spec = importlib.util.spec_from_file_location("v33", "scripts/analyze_rgbd_arms_confirm_v33.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
met = v32.is_file() and analysis.run_condition(json.loads(v32.read_text()))
(root / "run_condition.json").write_text(json.dumps({
    "condition": lock["run_condition"], "met": bool(met),
    "v32_results_sha256": hashlib.sha256(v32.read_bytes()).hexdigest() if v32.is_file() else None,
}, indent=2) + "\n")
raise SystemExit(0 if met else 3)
EOF
status=$?
if [ $status -eq 3 ]; then echo "V33_NOT_RUN $(date '+%F %T') condition not met"; exit 0; fi
[ $status -eq 0 ] || { echo "V33_LOCK_FAILED $(date '+%F %T')"; exit 1; }
core=48
for cohort in EVAL_V5 TRAIN_EXT REPLICA; do
  taskset -c $core .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_carrier_arms.py \
    --root "$V33" --stage seal --cohort $cohort > "$V33/audit/seal_$cohort.log" 2>&1 &
  core=$((core + 1))
done
wait
taskset -c 48 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/evaluate_rgbd_carrier_arms.py \
  --root "$V33" --stage score > "$V33/audit/score.log" 2>&1 || exit 1
taskset -c 48 .venv/bin/python ~/cv_tools/cv_host_run.py scripts/analyze_rgbd_arms_confirm_v33.py \
  > "$V33/audit/analysis.log" 2>&1
status=$?
echo "V33_DONE $(date '+%F %T') exit=$status"
exit $status
