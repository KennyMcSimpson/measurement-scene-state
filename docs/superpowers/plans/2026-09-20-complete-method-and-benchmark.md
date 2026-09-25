# Complete the approved inverse-JEPA and dynamic-TTT method

Authorized by Kenny on 2026-09-20: implement omitted code, train the complete model, compare against published standard-benchmark results, and iterate on non-core choices. No additional design approval is required for this scope. Existing V5 and all previous experiment artifacts remain preserved.

## Invariants

- Inverse-JEPA implementation: observed RGB/cameras build one query-independent typed scene state; a parameter-free measurement operator reads it; ground-truth training measurements supervise the state. No learned query-conditioned bypass enters the main method.
- Dynamic TTT: previous-state prediction, reveal current observation, compute pre-update feedback, prefix-only learned action selection, append evidence, trajectory-private fast write, rematerialize. ALL updates share one old state. No future/query labels in deployment features.
- Offline teachers use train-scene future/query labels only. B is frozen before labels are collected; changed carrier/write checkpoints invalidate old teacher/policy bindings.
- Develop and optimize using train/development data. A public final benchmark is not repeatedly tuned against as if it were validation; record any exploratory exposure explicitly and retain an untouched final evaluation.
- No new recurring automation, destructive cleanup, project-memory write, or baseline retraining by default.

## Required delivery (not an ablation program)

1. Implement structured prefix feedback, learned action values, feasible-action selection, versioned policy checkpoint and compute accounting.
2. Implement isolated offline action branches with horizon-averaged depth AbsRel, common masks, action advantages/costs, train-only provenance, deterministic initial roll-in and policy-driven recollection.
3. Fit action values with held-out training scenes; recollect on learned-policy prefixes and refit. Keep policy inputs free of future information.
4. Extend carrier/write training to configured model capacity, multiple train windows and OFF/fixed/mixed trajectories, preserving exact resume.
5. Integrate learned policy into evaluation with strict carrier/policy compatibility and save per-scene metrics/actions/timing.
6. Verify exact standard benchmark context/query/camera/resolution/metric protocol and public result source; implement its adapter and full-coverage checks. A subset or custom protocol cannot be labeled a standard full result.
7. Run resource checks, meaningful complete A/B/C/D training, development assessment, and standard evaluation when data/protocol permit.
8. If development performance is poor, diagnose concrete causes and change only capacity, optimization, sampling, numerical implementation or other non-core choices; keep every attempt/config and repeat. Do not promise superiority before evidence.

## Ownership

- Root: integration, resource configuration, launch/monitoring, comparison, non-core optimization decisions, independent verification.
- learned_policy: runtime feedback/control/policy/budget, policy checkpoint and fitting.
- action_teacher: train-only branch labeling, collection CLI and tests.
- training_scale: trainer configuration/action schedules, multi-window compiler and tests.
- benchmark_contract: official benchmark contract/data adapter and tests.
- completion_audit: read-only design-to-implementation gaps and final methodological review.

## Execution evidence

New experiments use `outputs/dynamic_full_method_20260920/` with separate integration, training, teacher, policy, development and benchmark directories. Each checkpoint and report records source/config/data hashes. The older 1,600-step run remains initial development evidence; it does not become full-method evidence by renaming.

Status at plan creation: work in progress; no complete-method training or standard-benchmark result exists yet.
