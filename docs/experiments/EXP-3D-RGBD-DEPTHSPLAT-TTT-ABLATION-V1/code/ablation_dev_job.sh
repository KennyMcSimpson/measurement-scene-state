#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1 development selection on DEV only (before its lock), run through
# cv_gpu_run.sh: (1) NV CTX/CACHE for c in 0.5, 1, 2 (opacity 0.95); (2) CPU scoring and choice of c by DEV ALL of
# NV_CACHE_F7; (3) NV TTT at the chosen c for m in 1, 3, 10; (4) CPU scoring and choice of m by DEV ALL of NV_TTT_F7.
set -u
w=~/cv_work/depthsplat/jepa_ttt
root=$w/ablation_dev
repo=/home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
V1=$repo/outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
DEVSEAL=$HOME/cv_work/depthsplat/probe_dev/raw/eval/DEV/prediction_seal.json
py=~/envs/depthsplat/bin/python
mkdir -p $root/geometry
cp -n $V1/frame_role_lock.json $root/
cp -n $w/rehearsal/geometry/geometry_seal_DEV.json $root/geometry/
score() {  # $@ = --ds specs; CPU only, project environment
  (cd $repo && CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 \
    .venv/bin/python $HOME/cv_tools/cv_host_run.py $V1/code/score_analyze.py $root score DEV --arms-seal $DEVSEAL --ds "$@")
}
cd ~/cv_work/depthsplat/depthsplat || exit 1
export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for c in 0.5 1 2; do
  $py $w/naive_eval.py $root DEV $root/ds/DEV/NVc$c --model NVc$c --scale-c $c --opacity 0.95 --arms CTX CACHE || exit 1
done
score NVc0.5=$root/ds/DEV/NVc0.5 NVc1=$root/ds/DEV/NVc1 NVc2=$root/ds/DEV/NVc2 || exit 1
mv $root/scores/DEV $root/scores/DEV_c
$repo/.venv/bin/python $w/ablation_select.py $root/scores/DEV_c/scene_table.json $root/selection_c.json \
  0.5="NVc0.5_CACHE_F7|ALL" 1="NVc1_CACHE_F7|ALL" 2="NVc2_CACHE_F7|ALL" || exit 1
C=$(python3 -c "import json; print(json.load(open('$root/selection_c.json'))['selected'])")
$py $w/naive_eval.py $root DEV $root/ds/DEV/NVsel --model NV --scale-c $C --opacity 0.95 --arms CACHE TTT \
  --ttt-mult 1 3 10 || exit 1
score NV=$root/ds/DEV/NVsel || exit 1
mv $root/scores/DEV $root/scores/DEV_m
$repo/.venv/bin/python $w/ablation_select.py $root/scores/DEV_m/scene_table.json $root/selection_m.json \
  1="NV_TTT_m1_F7|ALL" 3="NV_TTT_m3_F7|ALL" 10="NV_TTT_m10_F7|ALL" || exit 1
echo "ABLATION_DEV_DONE c=$C m=$(python3 -c "import json; print(json.load(open('$root/selection_m.json'))['selected'])")"
