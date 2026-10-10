#!/bin/bash
# After ft_v1: evaluate it, fine-tune ft_v2 (lower learning rate), evaluate ft_v2 -- each through cv_gpu_run.sh on GPU 4.
set -u
d=~/cv_work/depthsplat
until grep -q FINETUNE_DONE $d/ft_v1.log; do
  pgrep -u "$(id -u)" -f "[d]s_finetune.py $d/ft_v1" > /dev/null || { echo "ft_v1 ended without FINETUNE_DONE"; exit 1; }
  sleep 30
done
gpu() {  # retry while the GPU is taken (exit 75)
  while true; do ~/cv_tools/cv_gpu_run.sh --with-others --need-gib 24 -- "$@"; s=$?; [ $s -ne 75 ] && return $s; sleep 60; done
}
echo "$(date '+%F %T') eval ft_v1"
gpu bash $d/eval_ft_job.sh ft $d/ft_v1/depthsplat_ft_final.pth || exit 1
echo "$(date '+%F %T') train ft_v2"
gpu bash $d/ft_job.sh $d/ft_v2 --steps 4000 --eval-every 500 --lr 5e-6 > $d/ft_v2.log 2>&1 || { echo "ft_v2 failed"; exit 1; }
echo "$(date '+%F %T') eval ft_v2"
gpu bash $d/eval_ft_job.sh ft2 $d/ft_v2/depthsplat_ft_final.pth || exit 1
echo "$(date '+%F %T') CHAIN_DONE"
