# Evidence-Residual Scene State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible evidence-residual scene-state architecture and prepare an
official-split Hypersim expansion protocol.

**Architecture:** A feature pyramid samples calibrated context views at world-space voxel
hypotheses. Explicit cross-view agreement produces confidence and provenance. A bounded 3D
completion residual is gated by unknown probability before the unchanged fixed renderer reads the
final typed state.

**Tech Stack:** Python 3.11, PyTorch 2.7+, pytest, Ruff, YAML, existing MCSS CLI and manifests.

## Global Constraints

- Preserve all existing configs, checkpoints, outputs, camera conventions, and the 0.25 m A+ grid.
- Never pass target labels or target-camera queries into the state or completion encoder.
- Do not use the already observed Hypersim test subset for design or checkpoint selection.
- Keep at least 100 GiB free on D during any later dataset download.
- This directory has no `.git`; make a timestamped exact backup of modified source/config files
  instead of claiming commits.

---

### Task 1: Typed evidence contract and strict configuration

**Files:**
- Modify: `src/mcss/types.py`
- Modify: `src/mcss/model/system.py`
- Modify: `src/mcss/config.py`
- Test: `tests/test_types.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `StateEvidence` and optional `SceneState.evidence`.
- Produces: `ModelConfig.state_architecture`, `evidence_temperature`,
  `observed_residual_floor`, and `completion_residual_scale`.

- [ ] Add failing tests for evidence shape/range/provenance validation and `.to()` transfer.
- [ ] Run `uv run pytest tests/test_types.py -q` and verify failure because `StateEvidence` is absent.
- [ ] Implement `StateEvidence` and optional `SceneState.evidence` with strict validation.
- [ ] Run the focused type tests and verify they pass.
- [ ] Add failing config tests proving old configs default to `legacy` and new keys parse strictly.
- [ ] Implement optional model keys without making them required in legacy YAML.
- [ ] Run `uv run pytest tests/test_config.py -q` and verify it passes.

### Task 2: Multi-scale evidence and gated residual encoder

**Files:**
- Modify: `src/mcss/model/encoder2d.py`
- Create: `src/mcss/model/evidence_encoder.py`
- Modify: `src/mcss/model/system.py`
- Test: `tests/test_evidence_encoder.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: the Task 1 `StateEvidence` contract and `ModelConfig` keys.
- Produces: `EvidenceResidualStateEncoder.forward(context_rgb, cameras, bounds) -> SceneState`.

- [ ] Write a failing test for `_cross_view_evidence` showing identical valid observations have
  greater confidence than contradictory observations and invalid views have zero provenance.
- [ ] Run the focused test and verify it fails because the module is absent.
- [ ] Implement `PyramidImageEncoder` and `_cross_view_evidence`.
- [ ] Run the evidence helper test and verify it passes.
- [ ] Write a failing end-to-end test for evidence shapes, provenance sums, bounded residuals, and
  exact final-state reconstruction from base plus gated residual.
- [ ] Implement the evidence base state, residual trunk, zero-initialized heads, and model routing.
- [ ] Run `uv run pytest tests/test_evidence_encoder.py tests/test_model.py -q`.

### Task 3: Evidence regularization and telemetry

**Files:**
- Modify: `src/mcss/config.py`
- Modify: `src/mcss/engine.py`
- Test: `tests/test_config.py`
- Test: `tests/test_evidence_encoder.py`

**Interfaces:**
- Produces: `TrainingConfig.evidence_residual_weight`.
- Produces: `_state_regularization(state, evidence_residual_weight)` and
  `_state_diagnostics(state)`.

- [ ] Write failing tests for legacy regularization equivalence, high-confidence residual penalty,
  and evidence telemetry keys.
- [ ] Run focused tests and verify the expected failures.
- [ ] Implement evidence-aware regularization and telemetry while preserving legacy behavior.
- [ ] Add diagnostics to JSONL scalar records.
- [ ] Run focused engine/config tests and verify they pass.

### Task 4: New experiment configs and full-split planning

**Files:**
- Create: `configs/hypersim_er_smoke.yaml`
- Create: `configs/hypersim_er_overfit.yaml`
- Create: `configs/hypersim_er_train.yaml`
- Create: `configs/hypersim_er_val.yaml`
- Create: `configs/hypersim_er_diagnostic_test.yaml`
- Create: `configs/hypersim_er_final_holdout.yaml`
- Create: `scripts/plan_hypersim_expansion.py`
- Create: `tests/test_hypersim_expansion.py`
- Modify: `docs/DATASETS.md`
- Modify: `docs/PROTOCOL.md`
- Modify: `README.md`

**Interfaces:**
- Produces: deterministic CSV partitions from official metadata and an exclusion CSV.
- Produces: dry-run counts and byte-budget estimates without downloading.

- [ ] Write a failing fixture test proving official train/val scenes remain in their partitions and
  previously evaluated test scenes are excluded from final holdout.
- [ ] Implement the pure split-planning functions and CLI script.
- [ ] Run `uv run pytest tests/test_hypersim_expansion.py -q`.
- [ ] Add smoke, overfit, train, validation, diagnostic-test, and sealed-holdout configs.
- [ ] Document exact preparation and run commands plus the no-test-tuning boundary.
- [ ] Execute the planner in dry-run mode and record 365/46/20/26 scene counts.

### Task 5: Full verification

**Files:**
- Verify all modified and created files.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: fresh verification evidence and runnable commands.

- [ ] Run `uv run pytest -q` and require zero failures.
- [ ] Run `uv run ruff check .` and require zero errors.
- [ ] Compile every Python source in memory to avoid Windows path-length `.pyc` failures.
- [ ] Run a one-step CPU synthetic evidence-residual train/evaluate smoke.
- [ ] Run `uv run python scripts/cuda_smoke.py` after extending it to cover the new architecture.
- [ ] Run the main-resolution evidence-residual smoke and report peak CUDA memory.
- [ ] Do not start the long run until the overfit gate demonstrates sustained loss reduction.

