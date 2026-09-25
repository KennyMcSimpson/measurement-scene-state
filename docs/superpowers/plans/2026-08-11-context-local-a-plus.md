# Context-Local Metric A+ Protocol Plan

## Goal

Remove target/full-scene geometry from the state coordinate contract without changing the
Measurement-Complete Scene State hypothesis. The main model must still build one typed state
from calibrated context RGB and expose it only through the fixed measurement renderer.

## Compatibility boundary

- Existing configs keep `legacy_manifest_bounds` by default.
- Existing `outputs/hypersim_real_smoke` and `outputs/hypersim_pilot_*` artifacts are immutable.
- New A+ runs use `configs/hypersim_a_plus_*.yaml` and `outputs/hypersim_a_plus_*` only.
- Manifest bounds remain readable for legacy/oracle controls but are ignored by the A+ main path.

## Implemented contract

- [x] Anchor each sample to the first preselected context camera.
- [x] Transform context cameras, target query cameras, point maps, and world normals into the
  anchor frame with one rigid transform.
- [x] Use fixed metric local bounds from config; target labels cannot change them.
- [x] Build the query support mask from target camera rays and fixed bounds only.
- [x] Convert surfaces outside the local state into empty local-measurement labels without
  changing the query set.
- [x] Support anisotropic `[D, H, W]` grids.
- [x] Apply support consistently in losses and metrics; report query coverage and valid counts.
- [x] Report point error in meters, normalized point RMSE, and F-score at 0.05/0.25/0.50 m.
- [x] Add warmup-cosine scheduling, window means, EMA, epoch progress, LR, gradient norm, AMP
  scale/update status, scene IDs, validity ratios, timing, and CUDA memory to JSONL logs.
- [x] Use BF16 autocast for A+ after a preserved FP16 smoke exposed normal-gradient overflow.
- [x] Pass full unit tests, Ruff, formatting, compilation, synthetic smoke, and real CUDA smoke.

## Main A+ geometry

The current Hypersim A+ support is `x=[-6,6] m`, `y=[-4,4] m`, and `z=[0.05,12.05] m` in the
first context camera frame. The `[48,32,48]` state therefore has `0.25 m` spacing on every axis.
This is a local reconstruction task. It is not a claim that the state covers the complete source
scene.

A `24 m` value, if used in future frustum candidate construction, is a ray-distance cap. It must
not be described or implemented as the side length of a `24 m` cube.

## Execution gate

Run the two-step protocol smoke and two-step main-resolution smoke first, then the one-window
overfit experiment. Start the 50,000-step main run only after the overfit log shows a sustained
loss decrease and no optimizer/AMP skips. Validation and test must use the exact train checkpoint
with their matching A+ configs.

The failed FP16 diagnostic is preserved at `outputs/hypersim_a_plus_smoke`; its two updates were
skipped with non-finite normal gradients. The passing BF16 protocol and main-resolution evidence
is stored at `outputs/hypersim_a_plus_smoke_bf16` and
`outputs/hypersim_a_plus_mainres_smoke_bf16` respectively.
