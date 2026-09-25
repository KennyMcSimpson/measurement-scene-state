# Typed High-Resolution Appearance State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional doubled-resolution, evidence-residual RGB field while preserving the
accepted geometry state, fixed renderer, legacy checkpoints, and target-independent encoder.

**Architecture:** `StateAppearance` stores deterministic high-resolution evidence plus a bounded
completion residual. The evidence encoder constructs it only when `appearance_resolution_scale=2`,
and the existing renderer samples its reconstructed color for RGB while all other measurements keep
using native typed fields.

**Tech Stack:** Python 3.11, PyTorch 2.7+, pytest, Ruff, YAML, existing MCSS CLI.

## Global Constraints

- Do not pass target cameras or labels into `EvidenceResidualStateEncoder`.
- Keep geometry resolution `[48, 32, 48]`, fixed local bounds, and 64 ray samples unchanged.
- Preserve V4 configs, checkpoints, outputs, evidence fields, and all legacy behavior.
- Create only V5 output directories and retain the sealed final holdout.
- The directory has no Git repository; use the timestamped pre-change backup.

---

### Task 1: Typed Appearance And Strict Configuration

**Files:**
- Modify: `src/mcss/types.py`
- Modify: `src/mcss/model/system.py`
- Modify: `src/mcss/config.py`
- Test: `tests/test_types.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `StateAppearance` and optional `SceneState.appearance`.
- Produces: `ModelConfig.appearance_resolution_scale: Literal[1, 2]`.

- [ ] Add failing tests for appearance shape/range/provenance validation, exact color
  reconstruction, `.to()` transfer, legacy default scale one, explicit scale two, and rejection of
  scale two with the legacy architecture.
- [ ] Run the focused tests and verify failures identify the absent contract and config key.
- [ ] Implement `StateAppearance`, append `SceneState.appearance`, and parse the strict model key.
- [ ] Run the focused tests and require them to pass.

### Task 2: Fixed Renderer Appearance Sampling

**Files:**
- Modify: `src/mcss/measurements.py`
- Test: `tests/test_measurements.py`

**Interfaces:**
- Consumes: optional `SceneState.appearance`.
- Produces: RGB sampled from `appearance.color` without changing any other measurement.

- [ ] Add a failing renderer test with identical density and native color but a doubled-resolution
  red appearance field; assert RGB changes while depth is exact and appearance gradients are finite.
- [ ] Run the focused test and verify it fails because the renderer ignores appearance.
- [ ] Select the appearance color volume only at the `_sample_volume` RGB boundary.
- [ ] Run the focused renderer tests and require them to pass.

### Task 3: High-Resolution Evidence Completion

**Files:**
- Modify: `src/mcss/model/evidence_encoder.py`
- Modify: `src/mcss/model/system.py`
- Test: `tests/test_evidence_encoder.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `appearance_resolution_scale` and native completion hidden state.
- Produces: doubled deterministic evidence fields and bounded high-resolution color residual.

- [ ] Add failing tests for doubled shapes, exact reconstruction, zero residual initialization,
  evidence invariance after optimizer update, bounded residuals, finite backward, and scale-one
  absence.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement high-resolution projection/evidence construction and a compact lifted-hidden 3D
  residual head whose final convolution is zero initialized.
- [ ] Run focused encoder/model tests and require them to pass.

### Task 4: Regularization, Telemetry, And V5 Configs

**Files:**
- Modify: `src/mcss/engine.py`
- Modify: `scripts/cuda_smoke.py`
- Create: `configs/hypersim_er_v5_appearance2x_smoke.yaml`
- Create: `configs/hypersim_er_v5_appearance2x_mainres_smoke.yaml`
- Create: `configs/hypersim_er_v5_appearance2x_probe.yaml`
- Create: `configs/hypersim_er_v5_appearance2x_overfit.yaml`
- Create: `configs/hypersim_er_v5_appearance2x_train.yaml`
- Create: `configs/hypersim_er_v5_appearance2x_val.yaml`
- Modify: `README.md`
- Test: `tests/test_evidence_encoder.py`

**Interfaces:**
- Produces: appearance rewrite regularization and `state/appearance_*` JSONL telemetry.
- Produces: isolated smoke, probe, overfit, train, and validation commands.

- [ ] Add failing regularization/telemetry assertions for appearance precedence.
- [ ] Implement appearance-aware color rewrite penalty and scalar logs.
- [ ] Extend CUDA smoke with one scale-two fixed-model case.
- [ ] Add V5 configs with unchanged data protocol and isolated output directories.
- [ ] Document the exact smoke, probe, overfit, train, and validation commands.

### Task 5: Verification And Handoff

**Files:**
- Verify all modified and created files.

**Interfaces:**
- Produces: a runnable V5 overfit command, not a main-run recommendation.

- [ ] Run all focused tests, then `uv run pytest -q` with zero failures.
- [ ] Run `uv run ruff check .` with zero errors.
- [ ] Compile every Python file below `src` and `scripts`.
- [ ] Run CPU synthetic train/evaluate smoke and the complete CUDA smoke.
- [ ] Run the V5 main-resolution smoke and short real one-window probe.
- [ ] Inspect fresh logs for finite gradients, stable evidence, appearance residual activity, loss
  reduction, runtime, and peak memory.
- [ ] Stop before the 5,000-step overfit and give Kenny its exact command.
