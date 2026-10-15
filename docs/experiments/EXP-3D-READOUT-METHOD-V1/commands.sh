#!/bin/bash
# EXP-3D-READOUT-METHOD-V1 (Reliable Geometry Readout): the commands as they were run on hit, in order (2026-10-15).
# Repository: ~/zonghan_o_mirror/measurement-scene-state, branch research/reliable-geometry-readout-v1 (from
# research/caju-utility-v1 at 7607172, local only). The experiment root is outputs/EXP-3D-READOUT-METHOD-V1 ($E);
# docs/experiments/EXP-3D-READOUT-METHOD-V1 holds the committed copy (no caches, predictions or weights).
# GPU stages run through ~/cv_tools/cv_gpu_run.sh --with-others, which only ever picks GPU 4 (approved 2026-10-05).
set -eu
cd ~/zonghan_o_mirror/measurement-scene-state
E=outputs/EXP-3D-READOUT-METHOD-V1
DS=$HOME/envs/depthsplat/bin/python
git switch -c research/reliable-geometry-readout-v1

# Phase 0: metadata-only exposure audit (no image, depth or ground truth read) -> results/exposure_audit.json
CUDA_VISIBLE_DEVICES= .venv/bin/python $E/code/audit_exposure.py $E/results                 # 00:36

# Input caches from the sealed predictions (R7, near7, cosine, TTT and feed-forward renders; no ground truth)
CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=4 taskset -c 40-47 .venv/bin/python $E/code/rm_cache.py $E \
  > $E/audit/cache.log 2>&1                                                                   # RM_CACHE_DONE, 2040 queries

# Code smoke (train FIT_SYSTEM, score FIT_SYSTEM2 = a training cohort; 1 seed, 1 epoch) -> smoke/
~/cv_tools/cv_gpu_run.sh --with-others --need-gib 8 -- bash ~/cv_work/caju3d/rm_smoke.sh     # 00:44

# Preregistration lock (before any evaluation fold was scored), then the main run started at once
.venv/bin/python $E/code/rm_lock.py $E                          # 2026-10-15 00:46:10 +0800, sha256 6e6cda48...
setsid nohup ~/cv_tools/cv_gpu_run.sh --with-others --need-gib 12 -- bash ~/cv_work/caju3d/rm_run.sh &
#   Phase 1 + 2, leave one cohort out (4 folds x 6 network families x 3 seeds), 88 min on GPU 4
#   -> results/phase12/fold_*.json + preds_*.npz; audit/run.log: RM_RUN_DONE 5273.9 s, ground-truth reads 2040
# rm_stats.py was written after the lock, checked on the smoke fold, and its hash logged while results/phase12 was
# still empty: audit/stats_code_locked_at.txt  2026-10-15 00:47:48 +0800, sha256 4aecca4e...
CUDA_VISIBLE_DEVICES= .venv/bin/python $E/code/rm_stats.py $E smoke                         # 00:47

# Statistics, run twice; the second output must be byte-identical
CUDA_VISIBLE_DEVICES= taskset -c 40-47 .venv/bin/python $E/code/rm_stats.py $E
cp $E/BOOTSTRAP_RESULTS.json $E/audit/BOOTSTRAP_RESULTS.first_run.json
CUDA_VISIBLE_DEVICES= taskset -c 40-47 .venv/bin/python $E/code/rm_stats.py $E
cmp $E/BOOTSTRAP_RESULTS.json $E/audit/BOOTSTRAP_RESULTS.first_run.json && echo ANALYSIS_DETERMINISTIC > $E/audit/analysis_determinism.txt

# Runtime profile (no ground truth). A first single-process attempt failed inside the DepthSplat environment's R7
# path (audit/runtime.log, IndexError in frame_points); rerun as two parts in the environments the stages ran in:
cd $E/../.. && CUDA_VISIBLE_DEVICES= CV_ROOT=/home/xmum26_zonghan/zonghan_o_mirror OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src taskset -c 40-47 .venv/bin/python ~/cv_tools/cv_host_run.py $E/code/rm_runtime.py r7 $E \
  > $E/audit/runtime_r7.log 2>&1                                                             # 02:32, matches sealed R7
~/cv_tools/cv_gpu_run.sh --with-others --need-gib 8 -- bash ~/cv_work/caju3d/rm_runtime.sh   # gpu part, 02:34
CUDA_VISIBLE_DEVICES= .venv/bin/python $E/code/rm_runtime.py merge $E                       # -> RUNTIME_PROFILE.json
#   (only the merge's system table and note text were edited between the gpu / r7 runs and the final merge)

# Amendment 1 (post hoc, after the Gate A / B results): B1_FF / B7_FF, the KF table refit on the feed-forward render
~/cv_tools/cv_gpu_run.sh --with-others --need-gib 12 -- bash ~/cv_work/caju3d/rm_amend1.sh smoke   # training cohorts only
CUDA_VISIBLE_DEVICES= .venv/bin/python $E/code/rm_amend1_stats.py $E smoke
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $DS $E/code/rm_amend1.py $E lock   # 2026-10-15 02:45:51 +0800, sha256 fc72e572...
setsid nohup ~/cv_tools/cv_gpu_run.sh --with-others --need-gib 16 -- bash ~/cv_work/caju3d/rm_amend1.sh run &
#   -> results/amend1/ (fusion_tables_ff.json, kf_fit.json: the TTT refit reproduces all 64 frozen bins), 817 s
CUDA_VISIBLE_DEVICES= taskset -c 48-55 .venv/bin/python $E/code/rm_amend1_stats.py $E
cp $E/AMENDMENT_1_RESULTS.json $E/audit/AMENDMENT_1_RESULTS.first_run.json
CUDA_VISIBLE_DEVICES= taskset -c 48-55 .venv/bin/python $E/code/rm_amend1_stats.py $E
cmp $E/AMENDMENT_1_RESULTS.json $E/audit/AMENDMENT_1_RESULTS.first_run.json && echo ANALYSIS_DETERMINISTIC > $E/audit/amend1_determinism.txt

# Amendment 2 (post hoc, written before Amendment 1's results): Candidate B premise, oracle bound, CPU, saved predictions
CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=1 taskset -c 40-47 .venv/bin/python $E/code/rm_amend2.py $E smoke \
  > $E/audit/amend2_smoke.log 2>&1                                                            # FIT_SYSTEM2 only
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python $E/code/rm_amend2.py $E lock   # 2026-10-15 02:55:42 +0800, b19a0fd8...
CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=1 taskset -c 40-47 .venv/bin/python $E/code/rm_amend2.py $E run \
  > $E/audit/amend2_run.log 2>&1                                                              # -> AMENDMENT_2_RESULTS.json

# Amendment 3 (post hoc, written after Amendment 2's result, before Amendment 1's): Candidate A free-space premise, CPU
CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=8 taskset -c 40-47 .venv/bin/python $E/code/rm_amend3.py $E smoke \
  > $E/audit/amend3_smoke.log 2>&1                                                            # FIT_SYSTEM2 only, 529 s
#   (after the smoke and before the lock, one line changed: NaN pixel coordinates of points behind a frame are mapped
#    to -1 before the cast; those points are outside the frame and never used, so results cannot change)
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python $E/code/rm_amend3.py $E lock   # 2026-10-15 03:11:13 +0800, 5e7cbc3b...
CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=8 setsid nohup bash -c "taskset -c 40-47 .venv/bin/python \
  $E/code/rm_amend3.py $E run > $E/audit/amend3_run.log 2>&1; echo AMEND3_EXIT \$? >> $E/audit/amend3_run.log" &

# Integrity report (standard library only)
python3 $E/code/rm_integrity.py $E
