# Typed High-Resolution Appearance State Design

## Measured Decision

The converged V4 appearance oracle used the accepted one-window training checkpoint and produced:

- native shared color: 19.0412 dB PSNR, 0.6304 SSIM;
- native per-target-view color: 19.1580 dB PSNR, 0.6346 SSIM;
- doubled-resolution shared color: 22.1713 dB PSNR, 0.7796 SSIM.

The high-resolution oracle gains 3.13 dB over native shared color. Giving every target view its own
inadmissible color volume gains only 0.12 dB. All three tail-improvement fractions are below one
percent, so the comparison is not explained by one branch stopping earlier. The selected production
change is therefore spatial appearance resolution, not spherical harmonics or target-view-specific
state.

## Research Contract

The project still constructs one target-independent typed state from calibrated context RGB. The
target camera remains a query used only by the fixed renderer. Geometry and its evidence stay on the
existing `[48, 32, 48]` metric grid. Appearance may use a denser explicit grid because RGB texture
has a finer spatial bandwidth than 0.25 m geometry.

The high-resolution field is not an arbitrary latent tensor. It has the auditable decomposition:

```
appearance_color = sigmoid(
    logit(deterministic_base_color)
    + completion_gate * bounded_color_logit_residual
)
```

## Typed Contract

`StateAppearance` contains high-resolution `confidence`, `unknown_probability`,
`completion_gate`, `provenance`, `base_color`, and `color_logit_residual`. It validates finite
values, ranges, shapes, device consistency, `unknown = 1 - confidence`, and normalized-or-empty
provenance. Its `color` property reconstructs the final RGB volume exactly.

`SceneState.appearance` is optional and appended after existing fields so legacy positional
construction remains valid. When absent, the renderer reads `SceneState.color` exactly as before.
When present, only RGB samples `StateAppearance.color`; density, depth, normals, points, visibility,
uncertainty, and learned-control features retain their existing fields and resolution.

## Encoder

`appearance_resolution_scale` is a strict model configuration field with default `1` and supported
values `1` or `2`. Scale one constructs no new modules and preserves old state dictionaries. Scale
two is supported only by `state_architecture: evidence_residual`.

For scale two, voxel centers on the doubled grid are projected into every context view. The same
deterministic multi-scale RGB descriptors calculate high-resolution confidence and provenance.
Their provenance-weighted RGB is the immutable base color. No learned feature participates in
these evidence fields.

The native 3D completion hidden state is trilinearly lifted to the appearance grid and concatenated
with high-resolution base color, confidence, and unknown probability. A compact 3D refinement head
predicts only the bounded color-logit residual. The head starts at zero, so initialization exactly
reconstructs deterministic high-resolution evidence. The configured completion gate limits rewriting
in observed regions and supplies full capacity in unsupported regions.

## Interpretability And Regularization

- High-resolution confidence, provenance, unknown probability, and base color are invariant across
  optimizer steps for fixed context inputs.
- The renderer has no learned parameters and applies the reconstruction formula directly.
- High-confidence appearance residual magnitude replaces native color residual magnitude in the
  rewrite penalty when `StateAppearance` exists.
- Logs add appearance evidence, unknown, gate, residual magnitude, and spatial scale.
- Native evidence fields remain available for geometry audits and backward-compatible diagnostics.
- The target image and target camera never enter either state encoder branch.

## Compatibility And Runs

Existing configs default to scale one, existing checkpoints load unchanged, and V4 configs and
outputs remain immutable. New configs use `hypersim_er_v5_appearance2x_*` names. The main 150k run
remains blocked until a fresh V5 one-window overfit and a short multi-scene validation probe pass.

## Acceptance Gates

1. Legacy and V4 scale-one state dictionaries load without missing or unexpected keys.
2. Typed appearance validation, transfer, reconstruction, and renderer precedence tests pass.
3. Scale-two evidence fields are doubled on every axis, deterministic across an optimizer update,
   and residuals stay within the configured bound.
4. Fixed, learned-head, and free-decoder controls remain finite; the fixed renderer stays
   parameter-free.
5. Full pytest, Ruff, source compilation, CPU synthetic smoke, BF16 CUDA smoke, and real-data
   main-resolution smoke pass.
6. A short real one-window probe must reduce RGB loss before Kenny starts the fresh 5,000-step V5
   overfit run.
