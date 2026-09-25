# Learned Prefix Action Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, prefix-only learned action-value policy and its separately bound checkpoint/training path while preserving existing dynamic.v1 carrier checkpoints, fixed policies, and runner APIs.

**Architecture:** Runtime control data remains in `mcss.dynamic.types`; `control_to_features` is the sole ordered feature-schema encoder. `LearnedActionPolicy` owns a small MLP and masks infeasible actions at `choose(control, feasible_actions)`. Policy checkpoints are independent files carrying carrier/write content hashes, feature schema, render/budget/utility bindings, teacher hashes, and training-only normalization. Offline fitting consumes the shared all-actions-per-prefix JSONL rows and validates by scene split.

**Tech Stack:** Python 3.11, PyTorch, dataclasses, JSONL, pytest.

## Global Constraints

- Runtime policy features use only already observed image/state/camera/budget/action values; no future/query labels, paths, or teacher objects.
- Feature order and normalization are deterministic and versioned; all runtime features and scores must be finite.
- Existing `ControlInput` construction, fixed-policy behavior, dynamic.v1 payload shape, and fixed-policy budget charge remain compatible.
- Teacher rows contain all four actions in one row with `feasible_mask`, nullable infeasible losses/advantages, and shared metadata bindings outside rows.
- Policy checkpoints are composed with carrier/write checkpoints and reject mismatched expected bindings.
- CPU-only focused tests are required; no training run is launched here.

### Task 1: Runtime prefix control and deterministic feature schema

**Files:**
- Modify: `src/mcss/dynamic/types.py`
- Modify: `src/mcss/dynamic/feedback.py`
- Modify: `src/mcss/dynamic/policy.py`
- Test: `tests/dynamic/test_policy.py`

**Interfaces:**
- `ControlInput` keeps existing required fields and adds optional observed-prefix fields with deterministic defaults.
- `control_to_features(control: ControlInput) -> torch.Tensor` returns the fixed one-dimensional schema tensor.
- `LearnedActionPolicy.choose(control, feasible_actions) -> Action` masks infeasible actions, validates finite values, and resolves ties in `OFF,FUSE,COMPLETE,ALL` order.

- [ ] Add shape/finite checks and default-compatible optional control fields.
- [ ] Implement fixed feature names/version and tensor conversion from pooled image stats, 4x4 residual summaries, coverage/valid counts, camera change, fast stats, prior action, remaining budget, and remaining steps.
- [ ] Add the MLP value model, deterministic choice, and focused leakage/finite/tie/mask tests.

### Task 2: Causal runner feedback and policy-work accounting

**Files:**
- Modify: `src/mcss/dynamic/feedback.py`
- Modify: `src/mcss/dynamic/runner.py`
- Modify: `src/mcss/dynamic/budget.py`
- Test: `tests/dynamic/test_runner.py`
- Test: `tests/dynamic/test_budget.py`

**Interfaces:**
- `preupdate_feedback` continues returning old scalar keys and can add serializable current-prefix summaries.
- Runner calls `policy.choose(control, feasible_actions)` and records the exact serializable `control_input`.
- `WorkBudget.policy_units(policy)` charges fixed policies one unit and learned MLP inference from declared dimensions.

- [ ] Extend pre-update feedback with current image feature/residual summaries, valid/coverage counts, and camera delta without changing old scalar semantics.
- [ ] Build the full control input before cache append, pass the legacy two-argument policy signature, and log it.
- [ ] Reserve learned-policy work in mandatory future budget calculations while leaving fixed-policy charge at one.
- [ ] Test no future/query fields enter logs/features and minimum-budget feasibility remains conservative.

### Task 3: Independent policy checkpoint binding

**Files:**
- Create: `src/mcss/dynamic/policy_checkpoint.py`
- Test: `tests/dynamic/test_policy_checkpoint.py`

**Interfaces:**
- `save_policy_checkpoint(path, policy, *, binding, provenance) -> str` saves CPU state, architecture, normalization, schema, and binding.
- `load_policy_checkpoint(path, *, expected_binding=None, device='cpu') -> tuple[LearnedActionPolicy, dict]` rejects mismatches before loading.

- [ ] Define the policy-only schema and canonical binding validation for model content hash, feature schema, render/budget/utility protocols, and teacher dataset/source hashes.
- [ ] Preserve exact normalization/architecture/seed metadata and strict state loading.
- [ ] Add round-trip, mismatch, nonfinite, and dynamic.v1 non-mutation tests.

### Task 4: Teacher-row fitter and training script

**Files:**
- Create: `src/mcss/training/policy_fit.py`
- Create: `scripts/train_action_policy.py`
- Test: `tests/dynamic/test_policy_fit.py`

**Interfaces:**
- `fit_action_policy(rows, *, hidden_dim, seed, validation_scene_ids, ...)` trains action-value regression from the shared all-action row schema, retaining near ties and fitting normalization on training rows only.
- JSONL parser validates exact row fields and fixed action order; heldout validation is scene-based.
- CLI loads JSONL and writes the independent policy checkpoint plus provenance/binding metadata.

- [ ] Parse/validate rows with no per-action-row alternative and reject feature/schema/action mismatches.
- [ ] Fit deterministic feature normalization on training scenes only; regress all feasible action advantages/values with masked loss and report near-tie coverage.
- [ ] Add heldout-scene validation and exact seed/source metadata to the output path.
- [ ] Keep the script importable and add a CPU synthetic-row smoke test.

## Verification

- [ ] Run focused policy, checkpoint, budget, runner, and fitting tests with the project Python.
- [ ] Run the complete existing dynamic test subset to verify fixed behavior and checkpoint compatibility.
- [ ] Run `ruff check` on changed Python files when available.

