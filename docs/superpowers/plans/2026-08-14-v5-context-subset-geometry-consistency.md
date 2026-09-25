# V5 Context-Subset Geometry Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a detachable training-only V5 context-subset geometry consistency loss with strict
configuration, reproducible view deletion, auditable telemetry, and no model or inference changes.

**Architecture:** The normal four-view V5 forward remains the supervised path. A second shared V5
encoder call removes one context view, then inserts only its density residual into the detached
four-view evidence base. The unchanged fixed renderer compares counterfactual depth and visibility
against the detached four-view prediction on an evidence-defined overlap.

**Tech Stack:** Python 3.11, PyTorch 2.7+, pytest, Ruff, YAML, existing MCSS typed states and fixed
measurement renderer.

## Global Constraints

- Preserve `context RGB + calibrated cameras -> V5 typed state -> 2x appearance -> fixed renderer`.
- Add no learned module, model parameter, state-dict key, inference branch, or target-conditioned
  state input.
- Weight zero must skip the complete subset path.
- Nonzero use is restricted to fixed-mode, four-context V5 evidence-residual training.
- Do not add RGB, normal, point, target-label, or residual-magnitude losses.
- Do not run Hypersim training or tune the weight on validation scenes.
- This source directory has no `.git`; do not claim a branch, commit, or worktree.

---

### Task 1: Strict opt-in configuration and deterministic subset selection

**Files:**
- Modify: `src/mcss/config.py`
- Create: `src/mcss/context_subset_consistency.py`
- Modify: `tests/test_config.py`
- Create: `tests/test_context_subset_consistency.py`

**Interfaces:**
- Produces: `TrainingConfig.context_subset_geometry_weight: float = 0.0`.
- Produces: `drop_context_view(context_rgb, cameras, dropped_index) -> ContextSubset`.
- Produces: `ramped_context_subset_geometry_weight(max_weight, optimizer_step, warmup_steps)`.

- [ ] Add config tests proving the default is zero, `0.1` parses, negative values fail, and nonzero
  use fails outside fixed four-context `evidence_residual` experiments.
- [ ] Run the focused config tests and verify they fail because the new field is rejected.
- [ ] Add subset tests proving RGB and camera selection preserve retained order and reject an
  invalid deletion index.
- [ ] Run the subset tests and verify import failure because the new module is absent.
- [ ] Implement the strict field, cross-section validation, subset selection, and update-based ramp.
- [ ] Run both focused test files and require zero failures.

### Task 2: Counterfactual geometry and detached consistency loss

**Files:**
- Modify: `src/mcss/context_subset_consistency.py`
- Modify: `tests/test_context_subset_consistency.py`

**Interfaces:**
- Produces: `compute_context_subset_geometry(full_state, subset_state, full_predictions,
  target_cameras, renderer, residual_saturation_threshold, collect_audit_diagnostics=True) ->
  ContextSubsetGeometryResult`.
- `ContextSubsetGeometryResult` exposes `loss`, tensor `terms`, lightweight scalar `diagnostics`,
  log-step-only `audit_diagnostics`, and per-sample diagnostics for scene-aware logging.

- [ ] Add a failing real-renderer test proving a changed subset residual creates positive finite
  depth/visibility consistency loss, gradients reach only the subset residual, and deterministic
  evidence tensors receive no gradient.
- [ ] Add a failing zero-overlap test proving finite zero loss and zero subset gradient.
- [ ] Implement the shared/stable mask, four-view evidence-base counterfactual, detached teacher,
  metric-voxel-normalized Smooth L1 depth, Smooth L1 visibility, quantiles, saturation, and
  per-sample diagnostics.
- [ ] Run the focused tests and require zero failures.

### Task 3: Training integration, reproducibility, and telemetry

**Files:**
- Modify: `src/mcss/engine.py`
- Modify: `tests/test_context_subset_consistency.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: the Task 1 subset and ramp helpers and Task 2 geometry result.
- Produces: an independent CPU `torch.Generator`, conditional subset encoder/render path, weighted
  loss, JSONL context metadata, and checkpointed subset-generator state for enabled runs only.

- [ ] Add a failing trainer test proving weight zero never calls the subset helper and enabled mode
  logs the raw loss, effective weight, dropped/retained indices, and per-scene diagnostics.
- [ ] Implement the conditional training path without changing evaluation or model forward.
- [ ] Store and restore the independent generator state only when the mechanism is enabled.
- [ ] Use the next optimizer-update index for ramping so all micro-steps in one accumulated update
  share one effective weight.
- [ ] Run focused engine and CLI tests and require zero failures.

### Task 4: Frozen runnable configurations and operator boundary

**Files:**
- Create: `configs/synthetic_er_v5_context_subset_geometry_smoke.yaml`
- Create: `configs/hypersim_er_v5_context_subset_geometry_train_v1.yaml`
- Modify: `README.md`

**Interfaces:**
- Produces: a two-step CPU verification config and a V5-matched, unexecuted Hypersim proposal with
  weight `0.1`, warmup `3000`, four context views, and a new output directory.

- [ ] Add the synthetic smoke config with four context views and no real-data dependency.
- [ ] Add the Hypersim config by changing only output directory and the one new weight relative to
  `hypersim_er_v5_phase0_pilot_train_v3.yaml`.
- [ ] Document that the smoke is software verification, the Hypersim config is unexecuted, and V5
  remains the comparator until all empirical gates in the design pass.
- [ ] Load both configs through the strict parser.

### Task 5: Verification and scope audit

**Files:**
- Verify all files named above.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: fresh test, lint, compilation, smoke, and compatibility evidence.

- [ ] Run focused context/config tests.
- [ ] Run the enabled two-step CPU training smoke and inspect JSONL/checkpoint metadata.
- [ ] Run the full pytest suite and require zero failures.
- [ ] Run Ruff over `src`, `tests`, and `scripts` and require zero errors; record existing
  `third_party` and `backups` failures separately rather than modifying vendor/history code.
- [ ] Compile every Python source in memory.
- [ ] Compare `MeasurementCompleteSystem` and model state-dict keys before and after; confirm neither
  the model file nor its keys changed.
- [ ] Audit changed files against the design and report that no Hypersim experiment was started.
