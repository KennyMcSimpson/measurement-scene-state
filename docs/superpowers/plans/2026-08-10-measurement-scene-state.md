# Measurement-Complete Scene State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a directly runnable sparse-view scene-state training and evaluation project with
fixed measurements, matched learned controls, safe Replica/Hypersim data tools, and mechanism
diagnostics.

**Architecture:** Calibrated context images are unprojected into a typed dense voxel state. A
parameter-free volume renderer composes all primary outputs; learned-head controls reuse the same
state encoder. A versioned manifest isolates dataset-specific camera and file conventions.

**Tech Stack:** Python 3.11, PyTorch 2.7+ cu128, NumPy, Pillow, h5py, PyYAML, requests, pytest,
Ruff, and uv.

## Global Constraints

- Project root is exactly `<workspace-root>`.
- Do not download large datasets during implementation.
- Main input is sparse calibrated RGB; RGB-D context is diagnostic only.
- Main outputs are ray depth, point map, world normal, and visibility; RGB is auxiliary.
- Fixed renderer modules must contain zero trainable parameters.
- Default smoke settings must fit CPU memory and the 12 GB RTX 5070.
- Real benchmark claims require matched splits, context counts, resolution, queries, and metrics.

---

### Task 1: Typed data and camera contracts

**Files:**
- Create: `src/mcss/types.py`
- Create: `src/mcss/geometry.py`
- Test: `tests/test_types.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces: `Cameras`, `SceneBatch`, `SceneState`, `pixel_grid`, `generate_rays`,
  `project_world`, `intersect_aabb`, and `make_voxel_centers`.

- [ ] Write tests for tensor shapes, invalid matrices, OpenCV ray direction, projection roundtrip,
  and ray-box misses.
- [ ] Run `uv run --extra dev pytest tests/test_types.py tests/test_geometry.py -q`; expect import
  failure because `mcss.types` and `mcss.geometry` do not exist.
- [ ] Implement validated dataclasses and batched geometry without dataset-specific conventions.
- [ ] Re-run the two test files and require zero failures.

### Task 2: Fixed measurement program

**Files:**
- Create: `src/mcss/measurements.py`
- Test: `tests/test_measurements.py`

**Interfaces:**
- Consumes: `SceneState`, `Cameras`, `generate_rays`, and `intersect_aabb`.
- Produces: `FixedMeasurementRenderer.forward(state, cameras, measurements=None)` returning
  channel-first `rgb`, `depth`, `normal`, `point`, `visibility`, and `uncertainty` tensors.

- [ ] Test an analytic high-density slab, output shapes, finite values, differentiability, query
  permutation equivariance, and zero renderer parameters.
- [ ] Run the test and confirm it fails because the renderer is missing.
- [ ] Implement chunked grid sampling, transmittance integration, density-gradient normals, and
  measurement composition.
- [ ] Re-run measurement and geometry tests and require zero failures.

### Task 3: Scene-state encoder and matched controls

**Files:**
- Create: `src/mcss/model/encoder2d.py`
- Create: `src/mcss/model/state_encoder.py`
- Create: `src/mcss/model/system.py`
- Create: `src/mcss/model/controls.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `UnprojectiveStateEncoder`, `MeasurementCompleteSystem`,
  `LearnedHeadsControl`, `FreeDecoderControl`, and `build_model(config)`.

- [ ] Test that target labels are not accepted by model forward, all modes share state shapes,
  fixed mode calls the parameter-free renderer, and forward/backward gradients are finite.
- [ ] Run the test and confirm missing-module failure.
- [ ] Implement residual 2D features, projection/fusion moments, 3D residual refinement, typed
  heads, and matched learned controls.
- [ ] Re-run model, measurement, and geometry tests.

### Task 4: Synthetic and manifest datasets

**Files:**
- Create: `src/mcss/data/synthetic.py`
- Create: `src/mcss/data/manifest.py`
- Create: `src/mcss/data/collate.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `SyntheticSceneDataset`, `ManifestSceneDataset`, `write_manifest`,
  `validate_manifest`, and `collate_scene_batches`.

- [ ] Test deterministic analytic views, context/target disjointness, metric depth and normal
  agreement, manifest path containment, resize-aware intrinsics, and batch collation.
- [ ] Run the test and confirm missing-module failure.
- [ ] Implement analytic sphere/box rendering and versioned manifest loading.
- [ ] Re-run data tests and the model forward/backward test on a synthetic batch.

### Task 5: Replica and Hypersim tools

**Files:**
- Create: `src/mcss/data/download.py`
- Create: `src/mcss/data/replica.py`
- Create: `src/mcss/data/hypersim.py`
- Test: `tests/test_download.py`
- Test: `tests/test_real_adapters.py`

**Interfaces:**
- Produces: resumable `download_replica`, range-based `download_hypersim_subset`,
  `prepare_replica`, and `prepare_hypersim`.

- [ ] Test Range handling with a local HTTP fixture, atomic partial files, archive path rejection,
  known Replica part URLs, Hypersim URL validation, and tiny on-disk adapter fixtures.
- [ ] Run tests and confirm missing-module failure.
- [ ] Implement safe downloads and conversion to the common manifest without contacting the
  network in tests.
- [ ] Re-run downloader and adapter tests.

### Task 6: Losses, metrics, and mechanism diagnostics

**Files:**
- Create: `src/mcss/losses.py`
- Create: `src/mcss/metrics.py`
- Create: `src/mcss/diagnostics.py`
- Test: `tests/test_objectives.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Produces: `MeasurementLoss`, `MetricAccumulator`, `measurement_generalization_matrix`,
  `cross_context_consistency`, `intervention_specificity`, `query_equivariance`, and
  `no_bypass_audit`.

- [ ] Test perfect predictions, masked invalid depth, known normal angles, visibility confusion
  counts, intervention selectivity, and no-bypass failures.
- [ ] Run tests and confirm missing-module failure.
- [ ] Implement objectives and diagnostics with explicit sample counts and NaN-safe masking.
- [ ] Re-run objective and diagnostic tests.

### Task 7: Training, evaluation, and CLI

**Files:**
- Create: `src/mcss/config.py`
- Create: `src/mcss/engine.py`
- Create: `src/mcss/cli.py`
- Create: `configs/synthetic_smoke.yaml`
- Create: `configs/hypersim_pilot.yaml`
- Create: `configs/replica_pilot.yaml`
- Test: `tests/test_config.py`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: `load_config`, `Trainer`, `Evaluator`, and CLI commands `train`, `evaluate`,
  `diagnose`, `download`, and `prepare`.

- [ ] Test config rejection, one-step CPU training, checkpoint resume, evaluation JSON schema,
  and `--dry-run` download behavior.
- [ ] Run CLI tests and confirm missing-module failure.
- [ ] Implement deterministic DataLoaders, AMP, accumulation, atomic checkpoints, JSONL logs,
  and argument parsing.
- [ ] Re-run CLI smoke tests and all unit tests.

### Task 8: Reproducible handoff and verification

**Files:**
- Create: `README.md`
- Create: `docs/DATASETS.md`
- Create: `docs/PROTOCOL.md`
- Create: `scripts/setup.ps1`
- Create: `scripts/smoke.ps1`

**Interfaces:**
- Documents exact setup, smoke, download, preparation, training, evaluation, and matched-control
  commands with output locations.

- [ ] Run `uv sync --extra dev` and record the resolved environment.
- [ ] Run `uv run --extra dev pytest -q` and require zero failures.
- [ ] Run `uv run --extra dev ruff check .` and require zero errors.
- [ ] Run `uv run --extra dev ruff format --check .` and require zero formatting differences.
- [ ] Run the synthetic CLI train and evaluate commands on CPU and inspect their JSON artifacts.
- [ ] Run a CUDA forward/backward smoke when CUDA is available and report it separately.
- [ ] Re-read the design and verify every implemented/unsupported item is stated accurately.

