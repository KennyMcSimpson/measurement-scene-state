# V6 Phase 0 Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce immutable, provenance-complete V5 checkpoint, renderer-sampling, surface-evidence, appearance-oracle, and 16/8 multi-scene pilot artifacts before V6 changes are admitted.

**Architecture:** Keep the existing trainer and V5 model numerically unchanged. Add narrow CLI/output helpers around evaluation, a read-only evidence-audit module that samples the frozen state on target rays after inference, and deterministic label-blind scene selection. Every command writes to a new output directory and refuses an existing report.

**Tech Stack:** Python 3.11, PyTorch 2.7+, dataclasses, argparse, JSON, pytest, Ruff, uv, PowerShell.

## Global Constraints

- Authorized project root is `<workspace-root>`.
- Do not read or materialize the 26 final-holdout scenes.
- Target depth is available only inside post-forward audit code and never enters a model API.
- Preserve `legacy` and `evidence_residual` model construction and checkpoint loading.
- Never overwrite an existing evaluation, audit, oracle, training log, or checkpoint artifact.
- Formal V5/V6a validation audits use 256 samples per ray and exactly four windows per scene,
  selected by the lowest `sha256("v6-surface-audit:<scene_id>:<window_start>")` values. Persist and
  reuse the exact selected rows so outputs and labels cannot influence window choice.
- V5 checkpoint comparison uses the same seed, scene/window, image size, context/target views, bounds, and metric code.
- The directory is not a Git repository. Replace commit steps with a fresh test run plus SHA-256 inventory of changed files.

---

## File Structure

- Create `src/mcss/evidence_audit.py`: pure target-ray sampling, bin assignment, and aggregate statistics.
- Create `src/mcss/phase0_runner.py`: immutable checkpoint evaluation, renderer sweep, evidence audit, and pilot selection orchestration.
- Create `scripts/run_v6_phase0.py`: thin source-checkout entry point.
- Modify `src/mcss/cli.py`: allow an explicit immutable evaluation output directory.
- Modify `src/mcss/config.py`: add an optional label-blind scene allowlist to `DatasetConfig`.
- Modify `src/mcss/engine.py`: pass the scene allowlist to real datasets; do not change model/evaluation math.
- Modify `src/mcss/data/manifest_dataset.py`: filter manifests before entries are constructed.
- Create `tests/test_evidence_audit.py`: ray bins, invalid-ray counts, finite aggregate statistics, and target/model separation.
- Create `tests/test_phase0_runner.py`: output refusal, config provenance, renderer sweep layout, and deterministic pilot selection.
- Modify `src/mcss/appearance_oracle.py` and `src/mcss/oracle_runner.py`: preserve V4
  variants and add an oracle that optimizes the existing V5 typed-appearance grid without changing
  native density.
- Modify `tests/test_appearance_oracle.py` and `tests/test_appearance_oracle_runner.py`: typed-field
  selection, geometry invariance, shared/per-view variants, and report provenance.
- Modify `tests/test_cli_smoke.py`, `tests/test_config.py`, and `tests/test_manifest_dataset.py`: CLI/config/dataset regression coverage.
- Create `configs/hypersim_er_v5_phase0.yaml`: one-window immutable V5 diagnostic source config.
- Create generated pilot configs only after deterministic scene selection succeeds.

### Task 1: Immutable Evaluation Output Override

**Files:**
- Modify: `src/mcss/cli.py`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `load_config(Path) -> ExperimentConfig`, frozen dataclasses, `Trainer.evaluate`.
- Produces: `_with_evaluation_overrides(...) -> ExperimentConfig` and CLI options `mcss evaluate --output-dir PATH --n-samples N --ray-chunk-size N`.

- [ ] **Step 1: Write the failing CLI test**

Add a test that trains the existing synthetic smoke config, evaluates to a temporary explicit
directory, asserts `evaluation_report.json` exists there, then invokes the same command again and
expects `FileExistsError` before model construction can overwrite it.

```python
def test_evaluate_output_dir_is_explicit_and_immutable(tmp_path: Path) -> None:
    config_path = _synthetic_config(tmp_path, max_steps=1)
    checkpoint = _train_checkpoint(config_path)
    report_dir = tmp_path / "eval_step_000001_samples_8"

    assert main([
        "evaluate", "--config", str(config_path), "--checkpoint", str(checkpoint),
        "--output-dir", str(report_dir),
    ]) == 0
    assert (report_dir / "evaluation_report.json").is_file()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main([
            "evaluate", "--config", str(config_path), "--checkpoint", str(checkpoint),
            "--output-dir", str(report_dir),
        ])
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
uv run --extra dev pytest tests/test_cli_smoke.py::test_evaluate_output_dir_is_explicit_and_immutable -q
```

Expected: argparse rejects `--output-dir` or the report is written to the config output directory.

- [ ] **Step 3: Implement the minimal override**

Add `--output-dir`, `--n-samples`, and `--ray-chunk-size` only to `evaluate`. Before constructing
`Trainer`, replace the frozen nested training/model configs and refuse either existing report file.

```python
from dataclasses import replace

def _with_evaluation_overrides(
    config: ExperimentConfig,
    *,
    output_dir: Path | None,
    n_samples: int | None,
    ray_chunk_size: int | None,
) -> ExperimentConfig:
    model = replace(
        config.model,
        n_samples=config.model.n_samples if n_samples is None else n_samples,
        ray_chunk_size=(
            config.model.ray_chunk_size if ray_chunk_size is None else ray_chunk_size
        ),
    )
    if output_dir is None:
        return replace(config, model=model)
    resolved = output_dir.resolve()
    for name in ("evaluation.json", "evaluation_report.json"):
        if (resolved / name).exists():
            raise FileExistsError(f"refusing to overwrite existing evaluation report: {resolved / name}")
    return replace(
        config,
        model=model,
        training=replace(config.training, output_dir=str(resolved)),
    )
```

The `evaluate` branch loads once, applies the override when provided, then constructs `Trainer`.

- [ ] **Step 4: Run targeted and existing CLI tests**

Run:

```powershell
uv run --extra dev pytest tests/test_cli_smoke.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Record a file hash checkpoint**

Run:

```powershell
Get-FileHash -Algorithm SHA256 src\mcss\cli.py,tests\test_cli_smoke.py
```

Expected: two SHA-256 rows; retain them in the execution log.

### Task 2: Surface-Evidence Audit Core

**Files:**
- Create: `src/mcss/evidence_audit.py`
- Test: `tests/test_evidence_audit.py`

**Interfaces:**
- Consumes: `SceneState`, `Cameras`, target depth `[B,V,1,H,W]`, `generate_rays`, `intersect_aabb`, and `_sample_volume`.
- Produces: `audit_evidence(state, cameras, target_depth, target_visibility, n_samples=256) -> dict[str, object]`.

- [ ] **Step 1: Write failing pure bin-assignment tests**

```python
def test_depth_bins_use_half_max_voxel_edge_surface_band() -> None:
    distances = torch.tensor([[[1.0, 1.8, 2.0, 2.2, 3.0]]])
    target = torch.tensor([[[2.0]]])
    free, surface, behind = classify_depth_samples(distances, target, surface_band=0.25)
    assert free.tolist() == [[[True, False, False, False, False]]]
    assert surface.tolist() == [[[False, True, True, True, False]]]
    assert behind.tolist() == [[[False, False, False, False, True]]]

def test_flat_confidence_audit_returns_finite_counts() -> None:
    state, cameras, target_depth, visibility = _auditable_state()
    report = audit_evidence(state, cameras, target_depth, visibility, n_samples=16)
    assert report["schema_version"] == "mcss.evidence_audit.v1"
    assert report["bins"]["surface"]["count"] > 0
    assert report["bins"]["free"]["count"] > 0
    assert all(math.isfinite(v) for v in report["auroc"].values())
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
uv run --extra dev pytest tests/test_evidence_audit.py -q
```

Expected: import failure for `mcss.evidence_audit`.

- [ ] **Step 3: Implement fixed target-ray sampling**

Implement `classify_depth_samples` with strict free/behind comparisons and an inclusive surface
band. In `audit_evidence`, reject states without `StateEvidence`; generate target rays, intersect
the state AABB, sample exactly centered uniform distances, and sample only
`state.evidence.confidence`. Compute voxel edges from `state.bounds` and `state.spatial_shape`; the
surface band is half the largest edge. Valid audit rays require finite positive target depth,
target visibility greater than 0.5, and an AABB hit.

Return JSON-safe counts, mean/std/deciles for `free`, `surface`, and `behind`, plus independently
computed surface-vs-free and surface-vs-behind AUROC/AUPRC, Brier score, and ten-bin reliability
data. Keep per-window and per-scene records; aggregate the primary result as an equal-weight scene
macro and a deterministic seed-17 scene bootstrap confidence interval. Implement rank-based binary
AUROC and precision-recall integration locally so Phase 0 adds no dependency. Return `None`, not
NaN, when a class is empty, and include excluded-ray counts. Pooled ray samples are secondary
diagnostics because samples on one ray are correlated.

- [ ] **Step 4: Add invalid and no-evidence tests**

```python
def test_audit_rejects_targetless_state_and_counts_invalid_rays() -> None:
    state, cameras, depth, visibility = _auditable_state()
    with pytest.raises(ValueError, match="evidence"):
        audit_evidence(replace(state, evidence=None), cameras, depth, visibility)
    depth[..., 0, 0] = 0.0
    report = audit_evidence(state, cameras, depth, visibility, n_samples=16)
    assert report["rays"]["invalid_target_depth"] == 1
```

- [ ] **Step 5: Run targeted tests and lint**

Run:

```powershell
uv run --extra dev pytest tests/test_evidence_audit.py -q
uv run --extra dev ruff check src\mcss\evidence_audit.py tests\test_evidence_audit.py
```

Expected: tests and Ruff pass.

### Task 3: Phase 0 Runner and Renderer Sweep

**Files:**
- Create: `src/mcss/phase0_runner.py`
- Create: `scripts/run_v6_phase0.py`
- Test: `tests/test_phase0_runner.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, checkpoint paths, `Trainer`, `audit_evidence`.
- Produces: `evaluate_checkpoint`, `run_renderer_sweep`, `run_surface_audit`, and an atomic top-level `phase0_report.json`.

- [ ] **Step 1: Write failing layout and overwrite-refusal tests**

```python
def test_renderer_sweep_uses_unique_provenance_directories(tmp_path: Path) -> None:
    config, checkpoint = _trained_synthetic_case(tmp_path)
    report = run_renderer_sweep(config, checkpoint, tmp_path / "sweep", (8, 12))
    assert [item["n_samples"] for item in report["runs"]] == [8, 12]
    assert (tmp_path / "sweep" / "samples_0008" / "evaluation_report.json").is_file()
    assert (tmp_path / "sweep" / "samples_0012" / "evaluation_report.json").is_file()
    with pytest.raises(FileExistsError):
        run_renderer_sweep(config, checkpoint, tmp_path / "sweep", (8, 12))
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
uv run --extra dev pytest tests/test_phase0_runner.py::test_renderer_sweep_uses_unique_provenance_directories -q
```

Expected: import failure for `mcss.phase0_runner`.

- [ ] **Step 3: Implement immutable orchestration**

Use `dataclasses.replace` to set `model.n_samples` and `training.output_dir` per run. Refuse the
top-level report and every child evaluation report before doing work. Each run records checkpoint
absolute path/SHA-256, source config absolute path/SHA-256, resolved config, scene/window counts,
metrics, elapsed seconds, and device. Write JSON through a `.part` file plus `os.replace`.

`run_surface_audit` loads the checkpoint once, iterates the evaluation dataset without shuffle,
runs model forward before accessing target depth, and aggregates per-window audit records without
matching samples across scenes.

- [ ] **Step 4: Add a source-checkout CLI**

The script accepts:

```text
--config PATH --checkpoint PATH --output-root PATH
--checkpoint-sweep PATH [PATH ...]
--sample-counts 64 128 256
--surface-audit
```

It requires at least one requested action and returns nonzero on any overwrite refusal or missing
modality. It does not start training.

- [ ] **Step 5: Run runner tests and CLI help**

Run:

```powershell
uv run --extra dev pytest tests/test_phase0_runner.py -q
uv run python scripts\run_v6_phase0.py --help
```

Expected: tests pass and help lists all arguments.

### Task 4: V5 Typed-Appearance Oracle

**Files:**
- Modify: `src/mcss/appearance_oracle.py`
- Modify: `src/mcss/oracle_runner.py`
- Modify: `tests/test_appearance_oracle.py`
- Modify: `tests/test_appearance_oracle_runner.py`

**Interfaces:**
- Produces: CLI option `--field native|typed_appearance`; native preserves the existing V4 variants, typed appearance runs `typed_shared` and `typed_per_view` on the existing V5 appearance resolution.

- [ ] **Step 1: Write failing typed-field tests**

```python
def test_typed_appearance_oracle_keeps_native_geometry_and_optimizes_typed_color() -> None:
    state = _state_with_typed_appearance(native_resolution=4, appearance_resolution=8)
    oracle = AppearanceOracle(
        state, target_views=2, variant="typed_shared", field="typed_appearance"
    )
    assert oracle.spatial_shape == (8, 8, 8)
    torch.testing.assert_close(oracle.density_logits, state.density_logits)
    rgb = oracle.render_rgb(FixedMeasurementRenderer(n_samples=8), _two_cameras())
    rgb.mean().backward()
    assert oracle.color_logits.grad is not None
```

Add a runner test asserting `--field typed_appearance` refuses a state without
`StateAppearance`, writes field/shape/checkpoint hashes, and reports only shared/per-view typed
variants.

- [ ] **Step 2: Run tests and verify RED**

```powershell
uv run --extra dev pytest tests/test_appearance_oracle.py tests/test_appearance_oracle_runner.py -q
```

Expected: parser/variant rejection for `typed_appearance` and `typed_shared`.

- [ ] **Step 3: Implement typed shared/per-view states**

For `field="typed_appearance"`, initialize logits from `reference_state.appearance.color`; retain
native density, native color, log variance, and bounds. To render, build a `StateAppearance` whose
base color is 0.5, completion gate is one, residual is the optimized color logits, confidence and
provenance are zero, and unknown probability is one. Attach it to a native `SceneState`. For the
per-view variant, construct one such state per target view. Never upsample density or create a
second 2x grid beyond the existing typed appearance shape.

Keep `field="native"` behavior and report schema compatible. Add `field`, native state shape, and
optimized field shape to the report.

- [ ] **Step 4: Prove geometry invariance and report refusal**

Tests render depth before and after oracle color optimization and require exact equality within
dtype tolerance. Re-running into an existing `oracle.json` must raise `FileExistsError` without
truncating `oracle.jsonl`.

- [ ] **Step 5: Run targeted and existing oracle tests**

```powershell
uv run --extra dev pytest tests/test_appearance_oracle.py tests/test_appearance_oracle_runner.py -q
uv run --extra dev ruff check src\mcss\appearance_oracle.py src\mcss\oracle_runner.py tests\test_appearance_oracle.py tests\test_appearance_oracle_runner.py
```

Expected: all oracle tests and Ruff pass.

### Task 5: Deterministic 16/8 Pilot Scene Selection

**Files:**
- Modify: `src/mcss/config.py`
- Modify: `src/mcss/engine.py`
- Modify: `src/mcss/data/manifest_dataset.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_manifest_dataset.py`
- Modify: `src/mcss/phase0_runner.py`
- Test: `tests/test_phase0_runner.py`

**Interfaces:**
- Produces: `DatasetConfig.scene_ids: tuple[str, ...] | None`, dataset manifest filtering, and `select_pilot_scenes(root, split, count) -> tuple[str, ...]`.

- [ ] **Step 1: Write failing hash-selection and allowlist tests**

```python
def test_pilot_selection_is_label_blind_and_order_independent(tmp_path: Path) -> None:
    for scene_id in ("scene_c", "scene_a", "scene_b"):
        _write_minimal_manifest(tmp_path / scene_id, scene_id)
    first = select_pilot_scenes(tmp_path, split="train", count=2)
    second = select_pilot_scenes(tmp_path, split="train", count=2)
    assert first == second
    expected = tuple(sorted(
        ("scene_a", "scene_b", "scene_c"),
        key=lambda value: hashlib.sha256(f"v6-pilot:train:{value}".encode()).hexdigest(),
    )[:2])
    assert first == expected
```

Add config and dataset tests proving that unknown scene IDs fail and only selected manifests create
entries.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
uv run --extra dev pytest tests/test_phase0_runner.py tests/test_config.py tests/test_manifest_dataset.py -q
```

Expected: failures for missing `scene_ids` and selection function.

- [ ] **Step 3: Implement the allowlist**

Parse optional YAML `dataset.scene_ids` as a nonempty unique string list and store it as a tuple.
Pass it through `build_dataset` into `ManifestSceneDataset`. Filter loaded manifests by exact
`manifest.scene_id` before constructing entries. Raise `ValueError` listing requested IDs not found.
Omitted `scene_ids` preserves current behavior exactly.

Implement selection by discovering `manifest.json`, loading only each manifest's scene ID, sorting
by the specified SHA-256 key, and refusing `count` greater than available scenes.

- [ ] **Step 4: Write immutable selection artifacts and configs**

`phase0_runner` writes `pilot_selection.json` with input root, split, selection rule, candidate
count, selected IDs, and manifest hashes. Generate train and validation YAML by replacing only
`root`, `scene_ids`, `length`, `max_steps`, and `output_dir` from the accepted V5 config. The train
pilot uses 10,000 micro-steps; validation never trains.

- [ ] **Step 5: Run affected and full unit tests**

Run:

```powershell
uv run --extra dev pytest tests/test_phase0_runner.py tests/test_config.py tests/test_manifest_dataset.py -q
uv run --extra dev pytest -q
```

Expected: full suite passes.

### Task 6: Execute Phase 0 on Persisted V5 Artifacts

**Files:**
- Create: `configs/hypersim_er_v5_phase0.yaml`
- Create runtime artifacts only below: `outputs/hypersim_er_v6_phase0_*`

**Interfaces:**
- Consumes checkpoints `step_004000.pt`, `step_004500.pt`, and `step_005000.pt` from the V5 overfit output.
- Produces immutable reports used to decide V6a and renderer changes.

- [ ] **Step 1: Run pre-experiment verification**

```powershell
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run python -m compileall -q src scripts
uv run python scripts\cuda_smoke.py
```

Expected: all commands exit zero; CUDA smoke reports every current architecture/mode finite.

- [ ] **Step 2: Evaluate checkpoints and ray samples**

```powershell
uv run python scripts\run_v6_phase0.py `
  --config configs\hypersim_er_v5_phase0.yaml `
  --checkpoint outputs\hypersim_er_v5_appearance2x_overfit\checkpoints\step_004500.pt `
  --checkpoint-sweep `
    outputs\hypersim_er_v5_appearance2x_overfit\checkpoints\step_004000.pt `
    outputs\hypersim_er_v5_appearance2x_overfit\checkpoints\step_004500.pt `
    outputs\hypersim_er_v5_appearance2x_overfit\checkpoints\step_005000.pt `
  --sample-counts 64 128 256 `
  --surface-audit `
  --output-root outputs\hypersim_er_v6_phase0_v5_diagnostics
```

Expected: separate nonempty reports for each checkpoint and sample count, plus one surface audit.

- [ ] **Step 3: Re-run shared/per-view appearance oracle**

Use unique directories and the diagnostic step-4500 checkpoint. This is a training-window capacity
audit, not a final checkpoint-selection rule:

```powershell
uv run python scripts\run_appearance_oracle.py `
  --config configs\hypersim_er_v5_phase0.yaml `
  --checkpoint outputs\hypersim_er_v5_appearance2x_overfit\checkpoints\step_004500.pt `
  --output-dir outputs\hypersim_er_v6_phase0_oracle_step4500 `
  --field typed_appearance `
  --steps 1000 `
  --learning-rate 0.05
```

Expected: `oracle.json` records checkpoint/config hashes and typed shared/per-view variants while
native density remains fixed.

- [ ] **Step 4: Generate and run the fixed V5 pilot**

Generate selection/config artifacts, inspect that they contain 16 train and 8 validation IDs and
no test IDs, then run:

```powershell
uv run mcss train --config configs\hypersim_er_v5_phase0_pilot_train.yaml
uv run mcss evaluate `
  --config configs\hypersim_er_v5_phase0_pilot_val.yaml `
  --checkpoint outputs\hypersim_er_v6_phase0_v5_pilot_train\checkpoints\step_010000.pt `
  --output-dir outputs\hypersim_er_v6_phase0_v5_pilot_val_step10000
```

Expected: training and untouched validation reports reference only selected scene IDs.

- [ ] **Step 5: Seal the Phase 0 decision report**

Aggregate checkpoint replay, renderer delta, V5 per-scene surface-bin
ordering/AUROC/AUPRC/calibration, oracle gaps, and
pilot validation metrics into `outputs/hypersim_er_v6_phase0_decision.json`. Include a verdict for
each spec gate and SHA-256 for every source report. Explicitly state that evidence-audit metrics do
not select checkpoints. Do not modify V6 code until this report exists.

Run:

```powershell
Get-FileHash -Algorithm SHA256 outputs\hypersim_er_v6_phase0_decision.json
```

Expected: one SHA-256 row and a decision report with no missing gate.
