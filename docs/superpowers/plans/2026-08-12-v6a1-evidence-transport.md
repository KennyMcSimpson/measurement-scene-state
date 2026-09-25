# V6a.1 Fixed Ray-Evidence Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every
> behavior change and superpowers:verification-before-completion before any success claim.

**Goal:** Add a versioned, target-independent fixed ray-evidence transport architecture and run
its frozen one-window capacity gate.

**Architecture:** Reuse V6a's fixed local candidate scores, normalize by actual observer count,
require two independent observers, convert valid-only softmax excess into trilinearly transported
native-grid evidence, and keep the existing learned residual plus fixed renderer unchanged.

**Tech Stack:** Python 3.11, PyTorch 2.7+, pytest, Ruff, YAML configs, JSON/JSONL artifacts.

## Global Constraints

- Never alter V5 `evidence_residual` or V6a `dual_evidence` artifact semantics.
- No target label may enter evidence or state construction.
- No learned transport, learned renderer, target branch, image warp, Gaussian primitive, or new
  trainable parameter.
- Use seed 17 and the existing `48x32x48`, 4-context/2-target, 64-sample one-window protocol.
- This directory has no Git repository; use the versioned `backups/` snapshot and fresh outputs
  instead of commit steps.

---

### Task 1: Mass-conserving trilinear splat reducer

**Files:**
- Modify: `src/mcss/model/evidence_encoder.py`
- Test: `tests/test_evidence_encoder.py`

**Interfaces:**
- Produces `_trilinear_splat_candidates(points, values, valid, bounds, resolution) -> Tensor`.
- Input points are `[B,C,K,3]`; values/valid are `[B,C,K]`; output is `[B,N]`.

- [ ] Write tests where an exact voxel center receives all mass, an off-center point splits mass
      across eight cells, and an AABB-edge point preserves total mass after invalid-neighbor
      renormalization.
- [ ] Run `uv run --extra dev pytest tests/test_evidence_encoder.py -q` and confirm the new tests
      fail because the reducer is absent.
- [ ] Implement FP32 `scatter_add_` with center-coordinate conversion, eight fixed neighbor
      offsets, in-grid masking, and per-candidate weight renormalization.
- [ ] Re-run the focused tests and require all to pass.

### Task 2: Versioned transport evidence type and configuration route

**Files:**
- Modify: `src/mcss/types.py`
- Modify: `src/mcss/model/system.py`
- Modify: `src/mcss/config.py`
- Test: `tests/test_types.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces `TransportEvidence` with transported localization, transported appearance, transported
  peakness, coverage, V5 appearance fields, residuals, gates, and exact density/color properties.
- Adds `SceneState.transport_evidence` and architecture name `dual_evidence_transport`.

- [ ] Write failing tests for value ranges, `c_loc=c_app_transport*peak_transport`, reconstruction,
      `.to()`, mutually exclusive evidence payloads, YAML parsing, and unchanged V5/V6a defaults.
- [ ] Run the two focused test files and confirm expected failures.
- [ ] Add the dataclass, SceneState validation/transfer, architecture literal, config allowlist, and
      encoder routing without adding model parameters.
- [ ] Re-run focused tests and require all to pass.

### Task 3: Fixed transport encoder

**Files:**
- Modify: `src/mcss/model/evidence_encoder.py`
- Modify: `src/mcss/model/system.py`
- Test: `tests/test_evidence_encoder.py`

**Interfaces:**
- Produces `TransportEvidenceStateEncoder` and `_transport_surface_evidence(...)`.
- Consumes V5 appearance confidence and completion residuals; returns transported native-grid
  `R`, `A/R`, `M/A`, and `M/R` fields.

- [ ] Write failing tests for flat-profile zero mass, off-center peak displacement, two-view zero
      support, view-permutation invariance, detached evidence, unchanged parameter keys, and exact
      density reconstruction.
- [ ] Run focused tests and confirm failures are due to missing transport behavior.
- [ ] Implement the chunked reference/candidate loop with actual-observer score mean, minimum two
      observers, valid-only excess, and three splat accumulators.
- [ ] Keep color and high-resolution appearance exactly on the V5 path.
- [ ] Re-run focused tests and require all to pass.

### Task 4: Training diagnostics and evidence audit compatibility

**Files:**
- Modify: `src/mcss/engine.py`
- Modify: `src/mcss/evidence_audit.py`
- Modify: `src/mcss/evidence_telemetry.py`
- Test: `tests/test_evidence_encoder.py`
- Test: `tests/test_evidence_audit.py`
- Test: `tests/test_evidence_telemetry.py`

**Interfaces:**
- Engine logs transported support/peakness/coverage, gates, and rewrite energy.
- Surface audit selects `transport_evidence.surface_localization_support` for V6a.1.

- [ ] Write failing tests for regularization selection, diagnostics keys, audit field selection,
      and telemetry replay.
- [ ] Implement transport-specific branches without changing legacy/V5/V6a keys.
- [ ] Run all three focused test files and require all to pass.

### Task 5: Configs, static verification, and real-data gate

**Files:**
- Create: `configs/synthetic_v6a1_transport_smoke.yaml`
- Create: `configs/hypersim_er_v6a1_transport_overfit.yaml`
- Create runtime artifacts only below `outputs/hypersim_er_v6a1_transport_*`.

- [ ] Add isolated synthetic and Hypersim configs with seed 17, fixed renderer, unchanged capacity,
      `surface_peak_temperature: 0.05`, and a unique output directory.
- [ ] Run focused tests, full `pytest`, Ruff, and compileall.
- [ ] Run CPU synthetic train/evaluate and the existing CUDA smoke matrix extended with the new
      architecture; require finite outputs and zero renderer parameters.
- [ ] Train exactly 5,000 one-window steps, evaluate step 5000 in a fresh output directory, and
      write a gate report against V5 step 4500.
- [ ] If any capacity threshold fails, seal the failure and stop the branch. If all pass, run the
      frozen 32-window validation surface audit before considering V6b.

