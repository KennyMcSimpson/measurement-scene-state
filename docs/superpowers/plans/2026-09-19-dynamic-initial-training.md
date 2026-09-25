# Dynamic TTT initial sustained training and development evaluation

Authorized by Kenny: “那就开始训练开始跑测试”. Date: 2026-09-19.

## Purpose and scope

Run the first sustained optimization and development-set comparisons for the implemented dynamic carrier. This is an initial training baseline, not a convergence claim, final benchmark, or learned-controller result. The downloaded 26-scene final holdout is excluded. Existing V5 and previous failed/smoke artifacts are preserved. No recurring automation or project-memory write is created.

## Fixed first-run protocol

- Data: 365 train / 46 dev manifests were found. Strict fixed-16 compilation failed for `ai_011_010` (9 frames) and `ai_015_007` (6 frames). Before any main optimization, the run explicitly excludes these two insufficient-length train scenes and records them in the inventory, leaving **363 train / 46 dev episodes**. Use one sorted 16-frame window per scene: 4 warmup, 8 stream, 4 query. No bounds from labels. Other missing/invalid assets cause failure, never silent skipping.
- Preprocessing: existing 128x160 RGB and ray-distance metric depth; RGB-only online observations, RGB+depth offline training supervision. This is not strictly RGB-only supervision.
- Model: current small CarrierConfig (8^3 grid, 128 candidates, feature/hidden 8, expansion 16); no pretrained backbone. Keep fixed local metric bounds and parameter-free renderer, 16 samples per ray. Resource probe uses the same numerical settings.
- Seed: 20260919. Shuffle training scenes with a dedicated saved generator, log scene/query/action choices. Query selection and action selection must not accidentally stay tied to scene identity.
- Phase A: 1000 Adam steps, learning rate 0.001, gradient norm clip 1.0. Split the 12 available online frames into chronological even/odd contexts (6+6), independently build two OFF states, supervise both on one selected query view with the existing RGB+depth loss.
- Phase A anchor convention: each context uses its own first camera as its local coordinate anchor. The shared metric bounds therefore describe different world volumes for the two contexts. This is part of the initial training protocol and is not a controlled same-volume observation-subset ablation.
- Phase B: start from A-final; 600 Adam steps, learning rate 0.0003, clip 1.0. Train carrier and write rule jointly with full 4+8 observed unrolls; train fixed FUSE/COMPLETE/ALL trajectories without future inputs to write kernels. Offline query labels only enter the loss.
- Save phase-final inference checkpoints, periodic checkpoints, exact optimizer/RNG/cursor resume state, resolved configuration, source/data hashes, loss history and atomic status. Fail on nonfinite loss/gradients; do not silently change resolution or drop troublesome scenes.
- Evaluate a fixed untrained initialization, A-final OFF, and B-final OFF/FUSE/COMPLETE/ALL/RESIDUAL_THRESHOLD on dev. All B controls share the exact same checkpoint and views; threshold is a fixed untrained debug controller. Report per-scene RGB/depth metrics and scene-mean paired differences from OFF, plus time and work-proxy counts. This first comparison does not establish FLOP-matched fairness or prove learned dynamic selection.
- Evaluator preflight requires dev entries, unique scenes, 4 warmup/8 stream/4 query. For fixed policies, an actual-action mismatch is a failed comparison rather than a silently accepted OFF fallback. The full report additionally requires all 46 selected dev scenes.
- First run uses predetermined final steps, with no best-on-dev selection. Any subsequent changed training schedule is a distinct run with its own config and rationale.

## Execution

1. Independently prepare/validate training and dev episode artifacts.
2. Add minimal modular sustained trainer and checkpoint-aware evaluator with focused behavior tests.
3. Run a short CUDA resource probe and verify loss/gradients/checkpoint, without treating it as the main result.
4. Run and monitor A then B; run development evaluation on the frozen initial/A/B checkpoints.
5. Review results, independently reconstruct aggregate metrics, preserve V5, write a short Chinese result report with explicit limitations.

## Ownership

Dataset agent: compiler/tests/data indexes. Training agent: experiment training module/CLI/tests. Evaluation agent: checkpoint evaluation module/CLI/tests. Review agent: read-only methodology and implementation review. Root: protocol/config/resource probe, integration/launch/monitoring, independent verification and final report. Workers do not revert one another's edits.

## Interpretation boundaries

One short window per scene, one seed, and the tiny carrier limit representational capacity and statistical conclusions. Lower train loss alone does not show scene generalization. Gains between A and B can reflect additional carrier training; evidence for write benefit must compare B policies within the same checkpoint. Off-line training of Q/P/g is separate from per-scene deployment fast writes. Final holdout, learned policy, offline action teacher, direct-optimizer baselines and inverse-JEPA readout ablations remain later stages.
