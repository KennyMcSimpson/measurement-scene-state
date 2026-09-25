# Measurement-Complete Scene State

This repository implements the approved CV research direction:

> Multiple calibrated context RGB views are fused into one explicit typed scene state. A fixed,
> parameter-free measurement program reads that state for RGB, depth, normal, point map,
> visibility, and uncertainty, including measurement combinations withheld during training.

The main model is intentionally not a target-image predictor. It receives context RGB, context
camera matrices, fixed context-local bounds, and target camera queries only. The `learned_heads` and
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
The CUDA smoke checks forward and backward execution for all three model modes and exits with a
clear message when the installed PyTorch runtime cannot see a CUDA device.

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

The verified local Hypersim subset is split by scene. Train only from
`configs\hypersim_a_plus_train.yaml`; use `configs\hypersim_a_plus_val.yaml` and
`configs\hypersim_a_plus_test.yaml` with `mcss evaluate` and the train checkpoint. The old
`hypersim_pilot*.yaml` files retain their legacy manifest-bounds behavior for provenance.

Before a main run, execute the real-data plumbing smoke and one-window overfit gate:

```powershell
uv run mcss train --config configs\hypersim_a_plus_smoke.yaml
uv run mcss train --config configs\hypersim_a_plus_mainres_smoke.yaml
uv run mcss train --config configs\hypersim_a_plus_overfit.yaml
```

New JSONL logs include current, log-window mean, and EMA losses together with epoch progress,
learning rate, gradient norm, AMP scale/update status, scene IDs, modality validity, step time,
and peak CUDA memory.

## Current evidence boundary

The synthetic pipeline and CLI smoke are validated locally. Replica v1.0 assets and a
scene-disjoint 60/20/20 Hypersim subset are downloaded and structurally verified. Replica still
requires deterministic RGB-D trajectory rendering. A+ has passed two-step real-data protocol and
main-resolution BF16 CUDA smokes, but no real-data convergence or CVPR-level claim has been
established.
