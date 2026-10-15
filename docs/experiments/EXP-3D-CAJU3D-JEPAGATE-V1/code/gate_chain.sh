#!/bin/bash
# EXP-3D-CAJU3D-JEPAGATE-V1 chain (orchestration only): DEV GPU replay through cv_gpu_run (GPU 4; retry while no GPU),
# the DEV score + decision; the confirmation (GPU, then the twice-run analysis) only when decision.json says GO.
set -u
REPO=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
E=$REPO/outputs/EXP-3D-CAJU3D-JEPAGATE-V1
J=$E/code/gate_jobs.sh
LOG=$E/audit/chain.log
say() { echo "$(date "+%F %T") $*" | tee -a $LOG; }
gpu() {  # stage
  for attempt in $(seq 1 72); do
    ~/cv_tools/cv_gpu_run.sh --with-others --need-gib 28 -- bash $J "$1"
    rc=$?
    [ $rc -ne 75 ] && break
    say "no GPU (75), retry in 5 min"; sleep 300
  done
  say "$1 GPU rc=$rc"
  return $rc
}
cd $REPO
say GATE_CHAIN_START
gpu dev || exit 1
bash $J dev-score >> $LOG 2>&1
say "DEV_SCORE rc=$?"
if .venv/bin/python -c "import json, sys; sys.exit(0 if json.load(open('$E/decision.json'))['go'] else 1)"; then
  say "DEV said GO: confirmation"
  gpu conf || exit 1
  bash $J conf-score >> $LOG 2>&1
  say "CONF_SCORE rc=$?"
else
  say STOP_AT_DEV
fi
say GATE_CHAIN_END
