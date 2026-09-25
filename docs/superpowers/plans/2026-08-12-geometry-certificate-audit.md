# Geometry Certificate Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every
> behavior change and superpowers:verification-before-completion before any success claim.

**Goal:** Build and run a zero-training, label-blind audit of geometry-certified context
observations before changing or training the production scene-state model.

**Architecture:** A pure certificate builder consumes only context RGB and calibrated cameras. A
separate post-hoc evaluator loads context depth after certificate serialization, scores geometric
correctness, executes matched-coverage and shuffled controls, and writes immutable JSON plus a
Chinese-first Markdown report.

**Tech Stack:** Python 3.11, PyTorch 2.11/CUDA 12.8, Kornia 0.8 LoFTR, NumPy, Pillow, pytest,
Ruff, YAML, JSON.

## Global Constraints

- Preserve V5, V6a, V6a.1, the fixed renderer, and every prior output.
- Never expose target or context depth to certificate construction or threshold selection.
- Use only development-train scenes; do not read diagnostic-test or final-holdout data.
- Freeze seed 17 and every threshold in the design before post-hoc label scoring.
- Refuse to overwrite an existing audit directory or certificate artifact.
- This directory has no Git repository; do not manufacture commit history.

---

### Task 1: Geometry primitives and certificate contract

**Files:**
- Create: `src/mcss/geometry_certificates.py`
- Test: `tests/test_geometry_certificates.py`

- [ ] Write failing tests for DLT triangulation, cheirality, reprojection, reciprocal matches,
      cycle rejection, and immutable certificate serialization.
- [ ] Run the focused tests and confirm failure because the module is absent.
- [ ] Implement the smallest pure functions needed by the tests.
- [ ] Re-run the focused tests and require all to pass.

### Task 2: Label-blind proposer and certificate builder

**Files:**
- Modify: `src/mcss/geometry_certificates.py`
- Test: `tests/test_geometry_certificates.py`

- [ ] Write failing tests that a translated synthetic camera triplet recovers known 3D points,
      view permutation preserves sorted 3D certificates, and shuffled cameras fail certification.
- [ ] Implement frozen LoFTR pair proposals, epipolar distance, reciprocal/cycle checks, and
      multi-view geometric certification without any label argument.
- [ ] Run focused tests and require all to pass.

### Task 3: Post-hoc scorer, frozen runner, and report

**Files:**
- Create: `src/mcss/certificate_audit.py`
- Create: `scripts/run_geometry_certificate_audit.py`
- Create: `configs/hypersim_geometry_certificate_audit.yaml`
- Test: `tests/test_certificate_audit.py`

- [ ] Write failing tests for deterministic hash-window selection, label separation, decision
      rules, matched-coverage control, overwrite refusal, and Markdown rendering.
- [ ] Implement YAML parsing, manifest context loading, certificate-first serialization,
      depth-only-after-write evaluation, controls, aggregation, hashes, and terminal report.
- [ ] Run focused tests and require all to pass.

### Task 4: Static and real-data execution

**Files:**
- Runtime only: `outputs/geometry_certificate_audit_v1/**`
- Create terminal report: `outputs/geometry_certificate_audit_v1/REPORT.md`

- [ ] Run full pytest, Ruff, and compileall.
- [ ] Run a small deterministic smoke on one train scene and inspect stage counts and controls.
- [ ] Run exactly 32 frozen train windows and write certificates before loading depth labels.
- [ ] Apply the frozen decision rule and stop or authorize state integration without tuning.
- [ ] Verify hashes, artifact completeness, and no final/diagnostic partition access.
