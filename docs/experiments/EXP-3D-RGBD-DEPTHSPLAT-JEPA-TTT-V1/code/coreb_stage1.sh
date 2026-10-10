#!/bin/bash
# EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1, amendment 2: stage-1 scoring of Core B (TTT) with the locked score_analyze.py,
# after the ZS and ZS3 seals and only while the three reverse-JEPA trainings of amendment 1 are already running
# (the user asked for the TTT result first, then reverse JEPA). The J pipeline is fixed and automated (fixed step
# count, final step, no selection), so this query-depth read cannot influence it. J-dependent contrasts are NO_DATA
# here; the final scoring of every builder follows the J seals and writes scores/ and results/ afresh.
set -euo pipefail
cd /home/xmum26_zonghan/zonghan_o_mirror/measurement-scene-state
R=outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1
export CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1
RUN=".venv/bin/python $HOME/cv_tools/cv_host_run.py"
SEAL=$PWD/$R/raw/eval/FRESH78/prediction_seal.json
D=$PWD/$R/ds/FRESH78
M=$(python3 -c "import json; print(json.load(open('$R/ttt_selection.json'))['selected_mult'])")
running=$(ps -eo args | grep -c "[t]rain_jepa.py $PWD/$R/checkpoints/J" || true)
[ "$running" -eq 3 ] || { echo "ABORT: the three reverse-JEPA trainings are not running ($running)"; exit 1; }
[ ! -e $R/scores/FRESH78 ] && [ ! -e $R/stage1_coreB ] || { echo "ABORT: scores already exist"; exit 1; }
ls $D/ZS/ds_seal_0of3.json $D/ZS/ds_seal_1of3.json $D/ZS/ds_seal_2of3.json $D/ZS3/ds_seal_0of1.json > /dev/null
echo "STAGE1_START $(date '+%F %T') trainings_running=$running"
taskset -c 40-55 $RUN $R/code/score_analyze.py $PWD/$R score FRESH78 --arms-seal "$SEAL" --ds ZS=$D/ZS ZS3=$D/ZS3
taskset -c 40-55 $RUN $R/code/score_analyze.py $PWD/$R analyze FRESH78 --ttt-mult "$M"
taskset -c 40-55 $RUN $HOME/cv_work/depthsplat/jepa_ttt/posthoc_analysis.py $PWD/$R --ds ZS=$D/ZS ZS3=$D/ZS3
mkdir $R/stage1_coreB
mv $R/scores/FRESH78 $R/stage1_coreB/scores_FRESH78
mv $R/results/FRESH78_results.json $R/stage1_coreB/FRESH78_results_stage1.json
mv $R/results/FRESH78_posthoc_exploratory.json $R/stage1_coreB/FRESH78_posthoc_exploratory_stage1.json
echo "STAGE1_END $(date '+%F %T')"
