# Appearance Ceiling Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether native color optimization, spatial appearance resolution, or target
direction dependence explains the V4 one-window RGB ceiling.

**Architecture:** A reusable `AppearanceOracle` owns only optimizable color logits while borrowing
fixed density, variance, bounds, cameras, and the existing parameter-free renderer. A standalone
script loads the accepted V4 checkpoint, evaluates all three registered variants, and writes one
provenance-complete JSON report.

**Tech Stack:** Python 3.11, PyTorch 2.7+, pytest, existing MCSS dataset/config/checkpoint APIs.

## Global Constraints

- Use only `data/hypersim_prepared/train/ai_001_001` through the existing overfit config.
- Never pass target RGB to `MeasurementCompleteSystem.forward` or the state encoder.
- Never read validation, diagnostic-test, or final-holdout data.
- Keep the renderer parameter-free and accepted outputs immutable.
- The directory has no Git repository; create a timestamped backup before modifying existing files.

---

### Task 1: Oracle Variant Contract

**Files:**
- Create: `src/mcss/appearance_oracle.py`
- Create: `tests/test_appearance_oracle.py`

**Interfaces:**
- Produces: `OracleVariant`, `AppearanceOracle(reference_state, target_views, variant)`.
- Produces: `AppearanceOracle.render_rgb(renderer, cameras) -> Tensor`.

- [ ] Write tests proving native/shared, doubled/shared, and native/per-view parameter shapes; the
  reference state stays unchanged; per-view colors are routed to the matching cameras; and every
  variant backpropagates finite RGB gradients.
- [ ] Run `uv run pytest tests/test_appearance_oracle.py -q` and verify collection fails because
  `mcss.appearance_oracle` does not exist.
- [ ] Implement strict variant validation, detached fixed geometry buffers, logit initialization,
  high-resolution resampling, and per-view rendering through `Cameras.select_views`.
- [ ] Run `uv run pytest tests/test_appearance_oracle.py -q` and require all tests to pass.

### Task 2: Reproducible Oracle Runner

**Files:**
- Create: `src/mcss/oracle_runner.py`
- Create: `scripts/run_appearance_oracle.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `--config`, `--checkpoint`, `--output-dir`, `--steps`, `--learning-rate`, and `--seed`.
- Produces: `oracle.json`, `oracle.jsonl`, checkpoint/config SHA-256 values, and per-variant metrics.

- [ ] Add a CLI-level synthetic test that runs one step for all variants and asserts the report
  schema, source hashes, finite metrics, and distinct parameter counts.
- [ ] Run the CLI-level test and verify it fails because the runner is absent.
- [ ] Implement deterministic first-item loading, masked RGB MSE, best-state restoration, metric
  calculation through `compute_metrics`, atomic JSON output, and append-only progress records.
- [ ] Document the exact accepted-checkpoint command and diagnostic-only target-label boundary.
- [ ] Run the focused oracle tests and require them to pass.

### Task 3: Real V4 Diagnostic And Branch Decision

**Files:**
- Create: `outputs/appearance_oracle_v1/oracle.json`
- Create: `outputs/appearance_oracle_v1/oracle.jsonl`
- Create after measurement: `docs/superpowers/specs/2026-08-11-typed-appearance-state-design.md`
- Create after measurement: `docs/superpowers/plans/2026-08-11-typed-appearance-state.md`

**Interfaces:**
- Consumes: `configs/hypersim_er_overfit.yaml` and
  `outputs/hypersim_er_v4_scale4_overfit/checkpoints/step_005000.pt`.
- Produces: one selected production branch with measured justification.

- [ ] Run all three variants for enough steps that the last 20 percent does not improve best MSE
  by more than one percent, extending the run if this convergence check fails.
- [ ] Compare best PSNR/MSE against `native_shared`; select one branch using the design rule rather
  than inspecting any held-out data.
- [ ] Write the exact selected architecture, compatibility contract, telemetry, tests, smoke config,
  and overfit gate into the typed-appearance spec and plan.
- [ ] Self-review both documents for placeholders, contradictory tensor shapes, target leakage, and
  missing legacy compatibility.

### Task 4: Diagnostic Verification

**Files:**
- Verify all files created or modified above.

**Interfaces:**
- Produces: a measured and reproducible input to the next production implementation.

- [ ] Run `uv run pytest -q` and require zero failures.
- [ ] Run `uv run ruff check .` and require zero errors.
- [ ] Compile every Python file below `src` and `scripts` with `compile()`.
- [ ] Confirm `git status` reports that the directory is not a repository and report backup paths
  instead of claiming a commit.
