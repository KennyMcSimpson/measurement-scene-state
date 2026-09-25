# Appearance Ceiling Diagnostic Design

## Purpose

The V4 evidence-residual model fits one Hypersim window well in geometry but stops at 18.19 dB
RGB PSNR. Before changing the production architecture, a target-label oracle must separate three
possible ceilings: optimization of the current shared color grid, spatial resolution of that grid,
and view-independent appearance.

The oracle is diagnostic-only. It may optimize against hidden target RGB because its outputs are
never model predictions, checkpoints, baselines, or paper results. The production encoder remains
target-independent and the sealed final holdout remains untouched.

## Variants

All variants reuse density and bounds produced by the accepted V4 checkpoint and optimize only
color logits on the first deterministic overfit window.

- `native_shared`: one color volume at the current density-grid resolution, shared by all target
  cameras. This estimates how much of the gap is ordinary color optimization rather than missing
  representation.
- `highres_shared`: one shared color volume with every spatial axis doubled. Density and variance
  are fixed trilinear resamplings of the accepted state. This estimates the gain from separating
  appearance resolution from the 0.25 m geometry grid.
- `native_per_view`: one native-resolution color volume per target view. This is deliberately an
  inadmissible upper bound; its advantage over `native_shared` estimates the value available from
  target-direction-dependent appearance.

Every variant starts from the accepted state's color, uses the same fixed renderer and cameras,
optimizes masked RGB MSE, and reports initial/best/final PSNR, SSIM, MSE, parameter count, runtime,
and the checkpoint/config hashes.

## Selection Rule

Choose exactly one production change after the diagnostic.

1. If `highres_shared` gives the largest admissible gain, add an explicit high-resolution
   evidence-residual appearance field while keeping density, evidence confidence, and provenance
   on the existing geometry grid.
2. If `native_per_view` has a materially larger gain than `highres_shared`, add a bounded low-order
   directional color basis evaluated analytically by the parameter-free renderer. Per-view oracle
   volumes themselves are never admitted into the model.
3. If neither oracle materially exceeds `native_shared`, do not add a representation module;
   investigate encoder optimization and RGB objective conditioning instead.
4. If high-resolution and per-view gains are close, prefer high resolution first because it is an
   explicit spatial field and introduces fewer assumptions about reflectance.

The comparison is causal diagnostics, not a publication ablation. Exact observed deltas, not a
post-hoc preferred story, determine the branch.

## Safety And Compatibility

- Legacy and evidence-residual checkpoints keep loading without conversion.
- Target RGB is accepted only by the standalone oracle optimizer and never by model `forward`.
- The renderer remains parameter-free.
- Evidence confidence and provenance remain deterministic and optimizer-invariant.
- The diagnostic uses only the existing training overfit scene; validation, diagnostic test, and
  final holdout data are not read.
- Outputs are written below `outputs/appearance_oracle_v1` and cannot overwrite accepted runs.

## Acceptance

The diagnostic is accepted when focused unit tests pass, all three variants produce finite
gradients and JSON metrics on CUDA, the source checkpoint hash is recorded, and a rerun with the
same seed is deterministic within floating-point tolerance. Production work begins only after the
variant comparison identifies a single smallest justified change.
