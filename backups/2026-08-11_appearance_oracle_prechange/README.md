# Measurement-Complete Scene State

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

## Current evidence boundary

The evidence-residual V4 implementation passes CPU/CUDA forward-backward checks, a
main-resolution BF16 smoke, and a 5,000-step one-window overfit gate. That gate reaches depth AbsRel
0.00918, mean normal error 6.39 degrees, point F-score@0.25 m 0.9934, and RGB PSNR 18.19 dB while
keeping the evidence field exactly invariant across optimizer steps. The full 365-scene train,
46-scene validation, and 20-scene diagnostic-test partitions are downloaded, prepared, and
idempotently verified; the 26-scene final holdout remains unmaterialized and sealed. The 150,000-
microstep main run is intentionally not started because the one-window RGB ceiling still indicates
an appearance-representation bottleneck. No matched-baseline win or CVPR-level empirical claim has
been established yet.
