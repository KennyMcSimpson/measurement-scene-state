#!/bin/bash
# After chain_v2: scale-aligned injected variants (zero-shot and ft_v2), each through cv_gpu_run.sh on GPU 4.
set -u
d=~/cv_work/depthsplat
until grep -q -E "CHAIN_DONE|failed|without" $d/chain_v2.log; do sleep 30; done
grep -q CHAIN_DONE $d/chain_v2.log || { echo "chain_v2 did not finish"; exit 1; }
cat > $d/align_job.sh <<'JOB'
#!/bin/bash
set -u
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD
manifest=~/cv_work/depthsplat/probe_dev/manifest_dev.json
ft2=~/cv_work/depthsplat/ft_v2/depthsplat_ft_final.pth
run() {
  local name=$1; shift
  [ -e ~/cv_work/depthsplat/ds_dev_$name ] && { echo "SKIP $name (exists)"; return 0; }
  ~/envs/depthsplat/bin/python ~/cv_work/depthsplat/ds_predict.py "$manifest" DEV ~/cv_work/depthsplat/ds_dev_$name "$@" \
    > ~/cv_work/depthsplat/ds_dev_$name.log 2>&1 || { echo "FAILED $name"; return 1; }
  grep DS_PREDICTED ~/cv_work/depthsplat/ds_dev_$name.log
}
run zs_rgbd_x3_align --extra-rgb 3 --inject-measured-depth --align-extra-scale &&
run zs_rgbd_x9_align --extra-rgb 9 --inject-measured-depth --align-extra-scale &&
run ft2_rgbd_x9_align --extra-rgb 9 --inject-measured-depth --align-extra-scale --checkpoint "$ft2"
JOB
while true; do ~/cv_tools/cv_gpu_run.sh --with-others --need-gib 16 -- bash $d/align_job.sh; s=$?; [ $s -ne 75 ] && break; sleep 60; done
echo "$(date '+%F %T') ALIGN_DONE exit=$s"
