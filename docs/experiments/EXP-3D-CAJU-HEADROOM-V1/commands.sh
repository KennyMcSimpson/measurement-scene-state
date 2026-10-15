#!/bin/bash
# EXP-3D-CAJU-HEADROOM-V1 (CAJU-V1 Gate 0): the commands as they were run on hit, in order.
# Repository: ~/zonghan_o_mirror/measurement-scene-state, branch research/caju-utility-v1 (from main 4f0c0f6).
# CPU stages run through ~/cv_tools/cv_host_run.py with CUDA hidden; GPU stages through ~/cv_tools/cv_gpu_run.sh
# --with-others, which only ever picks GPU 4 (shared with another user's small resident process, approved 2026-10-05).
set -eu
E=outputs/EXP-3D-CAJU-HEADROOM-V1
J=$E/code/caju_jobs.sh

bash $J root          # 2026-10-14 01:0x  cohorts, episodes, contracts (metadata only)
bash $J geometry      # R7 at every query and history camera; CAJU_COMPAT replayed against V3 (max diff 0.0)
bash $J test          # 17 passed -> tests.json
~/cv_tools/cv_gpu_run.sh --with-others --need-gib 10 -- bash $J rehearsal   # 1 episode outside the root
bash $J lock          # 2026-10-14 01:07:17 +0800, preregistration.json sha256 f82179bf...

# Chain (orchestration only, ~/cv_work/caju/caju_chain.sh, copied to audit/caju_chain.sh):
#   cv_gpu_run.sh --with-others --need-gib 20 -- bash $J pilot     (CAJU_PILOT 2 shards + CAJU_COMPAT, then V1 references)
#   bash $J pilot-score                                           (prediction- and utility-level equivalence)
#   preregistered engine checks must all pass, else stop
#   cv_gpu_run.sh --with-others --need-gib 20 -- bash $J formal    (CAJU_SEL + CAJU_EVAL, 3 shards)
~/cv_work/caju/caju_chain.sh

# By hand, after the pilot report was reviewed:
bash $J score         # sel (fixed action, hash logged) -> eval -> analyze twice (byte-identical) -> finalize
