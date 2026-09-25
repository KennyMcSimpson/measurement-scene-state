# Measurement-Complete Scene State

> Current research baseline: V5 typed high-resolution appearance at step 4500. See
> [`BASELINE.md`](BASELINE.md). Later failed or exploratory branches remain preserved but do not
> replace this comparison anchor.

This repository implements the approved CV research direction:

> Multiple calibrated context RGB views are fused into one explicit typed scene state. A fixed,
> parameter-free measurement program reads that state for RGB, depth, normal, point map,
> visibility, and uncertainty, including measurement combinations withheld during training.

The main model is intentionally not a target-image predictor. It receives context RGB, context
camera matrices, fixed context-local bounds, and target camera queries only. Its evidence-residual
state separates deterministic cross-view RGB evidence from a bounded 3D completion residual and
keeps per-voxel confidence, unknown probability, and source-view provenance. The `learned_heads` and
`free_decoder` modes are matched controls, not the proposed method.

## Quick start

The project is isolated at `<workspace-root>`.

```powershell
cd <workspace-root>
uv sync --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run mcss train --config configs\synthetic_smoke.yaml
uv run mcss evaluate --config configs\synthetic_smoke.yaml `
  --checkpoint outputs\synthetic_smoke\checkpoints\step_000008.pt
uv run python scripts\cuda_smoke.py
```

The synthetic command is self-contained and does not download data. It writes checkpoints,
JSONL training logs, and evaluation JSON below `outputs/`, which is ignored by Git.
The CUDA smoke checks forward and backward execution for all three model modes under both the
legacy and evidence-residual state architectures. It exits with a clear message when the installed
PyTorch runtime cannot see a CUDA device.

## Commands

```text
mcss train --config CONFIG.yaml
mcss evaluate --config CONFIG.yaml --checkpoint CHECKPOINT.pt
mcss diagnose --config CONFIG.yaml --checkpoint CHECKPOINT.pt
mcss download replica --root data\replica_raw --dry-run
mcss download hypersim --root data\hypersim_raw --scenes ai_001_001 ai_001_002 --dry-run
mcss prepare replica --raw-root data\replica_rendered --output-root data\replica_prepared
mcss prepare hypersim --raw-root data\hypersim_raw --output-root data\hypersim_prepared --scenes ai_001_001
```

`docs/DATASETS.md` contains the official sources, disk estimates, subset workflow, and the
Replica rendering boundary. `docs/PROTOCOL.md` defines matched comparisons, held-out measurement
tests, metrics, and mechanism diagnostics.

The old verified 60/20/20 Hypersim subset and A+ configs remain diagnostic provenance. The new
protocol plans all 365 official train scenes, all 46 validation scenes, the 20 previously observed
test scenes as diagnostic-only, and the remaining 26 test scenes as a sealed final holdout. Inspect
and materialize the expansion with:

```powershell
uv run python scripts\plan_hypersim_expansion.py --dry-run
uv run python scripts\plan_hypersim_expansion.py
uv run python scripts\materialize_hypersim_expansion.py `
  --stage all --partitions train val diagnostic_test --workers 2
```

The materializer is resumable, preserves at least 100 GiB free by default, and refuses to touch
`final_holdout` without the explicit `--allow-final-holdout` release flag.

Before a main run, execute the real-data plumbing smoke and one-window overfit gate:

```powershell
uv run mcss train --config configs\hypersim_er_smoke.yaml
uv run mcss train --config configs\hypersim_er_mainres_smoke.yaml
uv run mcss train --config configs\hypersim_er_overfit.yaml
```

New JSONL logs include current, log-window mean, and EMA losses together with epoch progress,
learning rate, gradient norm, AMP scale/update status, scene IDs, modality validity, step time,
peak CUDA memory, evidence/unknown means, completion-gate use, and density/color residual energy.
Only after the overfit loss shows sustained reduction should the 150,000-microstep main config be
started with `configs\hypersim_er_train.yaml`.

To locate an RGB representation ceiling before changing the production model, run the
diagnostic-only appearance oracle against the accepted one-window checkpoint:

```powershell
uv run python scripts\run_appearance_oracle.py `
  --config configs\hypersim_er_overfit.yaml `
  --checkpoint outputs\hypersim_er_v4_scale4_overfit\checkpoints\step_005000.pt `
  --output-dir outputs\appearance_oracle_v1 `
  --steps 1000 `
  --learning-rate 0.05
```

This command intentionally optimizes hidden target RGB labels to estimate three unattainable
capacity ceilings: native shared color, doubled-resolution shared color, and one native color
volume per target view. Its outputs are diagnostics only. They are not model results, must not be
compared with baselines, and cannot be used on validation or test partitions.

The converged oracle selected doubled spatial appearance resolution: native shared color reached
19.04 dB, native per-view color 19.16 dB, and doubled-resolution shared color 22.17 dB. V5 therefore
adds only a typed high-resolution evidence-residual appearance field; it does not add a directional
or target-view-specific state. Run its gates in order:

```powershell
uv run mcss train --config configs\hypersim_er_v5_appearance2x_smoke.yaml
uv run mcss train --config configs\hypersim_er_v5_appearance2x_mainres_smoke.yaml
uv run mcss train --config configs\hypersim_er_v5_appearance2x_probe.yaml
uv run mcss train --config configs\hypersim_er_v5_appearance2x_overfit.yaml
uv run mcss evaluate --config configs\hypersim_er_v5_appearance2x_overfit.yaml `
  --checkpoint outputs\hypersim_er_v5_appearance2x_overfit\checkpoints\step_005000.pt
```

Do not start `configs\hypersim_er_v5_appearance2x_train.yaml` until the fresh 5,000-step V5
overfit result is evaluated. Its validation companion is
`configs\hypersim_er_v5_appearance2x_val.yaml`.

### V5 context-subset geometry consistency

The approved next direction is an opt-in training-only consistency path attached directly to the
unchanged V5 evidence-residual state. It removes one of the four context views, reuses the same
encoder and fixed renderer, and compares only counterfactual depth and visibility against a
detached full-context prediction on evidence-supported overlap. It does not add a model module,
state-dict key, target-label input, RGB consistency term, or inference branch. The strict design and
gates are in
[`docs/superpowers/specs/2026-08-14-v5-context-subset-geometry-consistency-design.md`](docs/superpowers/specs/2026-08-14-v5-context-subset-geometry-consistency-design.md).

The default `context_subset_geometry_weight` is zero, so existing V5 configs retain the original
training path. Run only the self-contained CPU software smoke while validating the implementation:

```powershell
uv run mcss train --config configs\synthetic_er_v5_context_subset_geometry_smoke.yaml
```

`configs\hypersim_er_v5_context_subset_geometry_train_v1.yaml` is a frozen, unexecuted **10,000
micro-step diagnostic pilot**, not the 150,000-step main run. It matches the V5
four-context/two-target Hypersim protocol with weight `0.1` and the existing 3,000-update warmup.
Do not start it until the design's one-window capacity, held-out context-drop, and eight-scene
validation gates are predeclared and evaluated. A lower consistency loss alone is not evidence
that it should replace V5.

## Current evidence boundary

The new geometry-certificate frontend has completed its pre-registered zero-training audit and is
sealed as `STOP_CERTIFICATE_FRONTEND`. Across 32 hash-selected Hypersim development-train scenes,
its certified source-grid coverage was 2.60%, median post-hoc 3D error was 1.52 m, p90 error was
12.24 m, and view-permutation point delta was 1.10 cm. These fail the frozen 15%, 0.25 m, 0.50 m,
and numerical-invariance gates. Do not attach this exact LoFTR plus reciprocal/cycle/DLT frontend
to the state or train it. The result and full report are under
`outputs/geometry_certificate_audit_v1/`; the failure seal is
`docs/superpowers/specs/2026-08-12-geometry-certificate-frontend-failure-seal.md`.

This failure does not alter the original context-to-typed-state-to-fixed-renderer backbone. It
rules out the current sparse certificate inlet. It blocked further code changes until a new
direction was discussed and approved; the 2026-08-14 V5 context-subset design discharges that
approval checkpoint without reopening the failed certificate branch.

The approved V7 branch qualifies a frozen MapAnything model as an external observation supplier.
It reads only context RGB and known cameras, seals structured supplier and certificate caches,
and opens context depth only for post-hoc evaluation. It does not modify the typed state, renderer,
or training loop. Both MapAnything and its DINOv2 architecture-code dependency are pinned and
verified locally; independent DINOv2 weights and remote hub loading are forbidden. Run the pinned
audit with:

```powershell
uv sync --extra dev --extra mapanything
uv run python scripts\run_mapanything_geometry_audit.py `
  --config configs\hypersim_mapanything_geometry_audit.yaml
```

The frozen contract is in
`docs/superpowers/specs/2026-08-12-v7-mapanything-certified-observation-design.md`.

The accepted V4 geometry checkpoint reaches depth AbsRel 0.00918, mean normal error 6.39 degrees,
point F-score@0.25 m 0.9934, and RGB PSNR 18.19 dB. The V5 implementation keeps that geometry and
adds a doubled-resolution typed appearance field selected by the training-only oracle. Its 200-step
real-data probe reduces the RGB training term from 0.1776 at step 10 to 0.0795 at step 200, versus
0.0932 for V4 at the same step, while keeping native and appearance evidence exactly invariant.
This is a capacity probe, not a final metric. The fresh V5 5,000-step overfit remains the next gate.
The full 365-scene train, 46-scene validation, and 20-scene diagnostic-test partitions are prepared;
the 26-scene final holdout remains unmaterialized and sealed. No main-run, matched-baseline win, or
CVPR-level empirical claim has been established yet.
