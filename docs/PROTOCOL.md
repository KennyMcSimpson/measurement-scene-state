# Experimental Protocol

## Main setting

- Static, calibrated sparse-view reconstruction.
- Context is 2 or 4 RGB images; target RGB is withheld at inference.
- Target queries use known intrinsics and camera-to-world matrices.
- The main A+ state is anchored to the first preselected context camera and uses fixed metric
  local bounds from the experiment config.
- Main outputs are metric ray depth, anchor-frame normal, anchor-frame point map, and local
  visibility. RGB is auxiliary and is reported separately.

## Spatial protocol

`context_local_metric` is the main protocol. Context and target cameras are rigidly expressed in
the first context camera frame. The anchor is chosen before target labels are loaded. The fixed
local bounds determine state extent, voxel spacing, ray sampling, and the query-only support mask.
Target RGB, depth, normal, point, visibility, and manifest bounds cannot affect those choices.

Target depth or point labels are used only after the query and support are fixed. If the observed
surface lies outside the local state, its local visibility label is zero, geometry labels are
invalidated, and auxiliary RGB uses the renderer background. The query remains in the protocol.
Report both query support coverage and the fraction of locally visible target surfaces so local
cropping cannot be hidden as an accuracy gain.

`legacy_manifest_bounds` is retained for reproducing old pilots and as a privileged/oracle
control. Prepared manifest bounds may contain geometry aggregated from every frame and therefore
cannot be used by the main A+ result.

The Hypersim A+ config uses a `12 x 8 x 12 m` local support and a `[48,32,48]` grid, which gives
`0.25 m` spacing on every axis. A future `24 m` candidate range denotes maximum ray distance, not
a `24 m` cube side length. A+ uses BF16 autocast on supported CUDA devices because the density-
gradient normal path can overflow FP16 dynamic loss scaling on nearly flat initial states.

### Hypersim camera and normal consistency check

The prepared validation split was checked by backprojecting metric ray depth, taking central
differences of neighboring world points, and comparing the resulting surface-normal axis with the
provided world normal. The check used three frames from each of 20 validation scenes and retained
924,413 locally continuous interior pixels. A horizontal-FOV sweep of 50/55/60/65/70 degrees gave
median angular errors of 3.79/2.22/1.76/3.08/4.88 degrees, respectively. At 60 degrees, the signed
cosine against `cross(dP/dx, dP/dy)` had median -0.99951; the negative sign is expected because
OpenCV image y points down while visible outward normals face the camera.

This is empirical support that the current 60-degree preview-camera assumption, metric ray depth,
pose conversion, and world normals are mutually consistent on the downloaded subset. It is not a
claim that the raw subset contains an explicit per-camera intrinsics artifact; the FOV assumption
must remain recorded and configurable.

### Cell-centered state semantics

Typed state values live at voxel cell centers. World points are therefore sampled with
`align_corners=False`, and density finite differences use cell sizes `extent / [W,H,D]`. Checkpoints
trained with the earlier boundary-aligned sampler are retained as diagnostic evidence but cannot be
combined with this renderer for final comparisons. All corrected A+ configs write to separate
`outputs/hypersim_a_plus_v2_cell_centered_*` directories so old and new runs cannot be mixed.

## Evidence-residual state

The proposed state architecture is

```text
final_state = evidence_base_state + completion_gate * bounded_completion_residual
```

All terms are 3D fields constructed before a target camera is queried. Cross-view feature/RGB
agreement over deterministic multi-scale RGB descriptors determines confidence and source-view
provenance. The learned image pyramid is reserved for completion features, so an optimizer update
cannot lower evidence confidence to obtain a larger residual gate. Unknown probability is one
minus confidence and controls completion capacity; a small residual floor permits correction in
observed regions. This is not target-image residual warping, and RGB-only input is not claimed to
directly observe free space.

The JSONL log reports mean confidence, unknown probability, completion gate, and raw density/color
residual magnitude. High-confidence residual rewriting is explicitly penalized. These fields make
the evidence/completion decomposition auditable, but the learned feature pyramid and 3D residual
network are not claimed to be fully human-interpretable.

## Measurement-complete claim

Define a training measurement set `M_train` and an evaluation set `M_eval`. A measurement
generalization matrix reports every pair. The key row trains the same fixed state with RGB/depth
and evaluates normal, point, and visibility without adding a target-specific head. Report the
held-out gap against a seen measurement and the corresponding controls.

The primary controls are:

1. `fixed`: explicit state plus parameter-free renderer.
2. `learned_heads`: same context encoder/state capacity, separate learned per-measurement heads.
3. `free_decoder`: same state features plus query origin/direction and a shared learned decoder.

External references such as SRT/OSRT, pixelNeRF, pixelSplat, MVSplat, DUSt3R, MASt3R, VGGT, and
per-scene 3DGS must be rerun or evaluated under exactly the same split, context count, camera
inputs, image resolution, query frames, and metric implementation. Original paper numbers are
not directly comparable to this protocol.

## Metrics

- RGB: PSNR, SSIM, optional LPIPS.
- Depth: AbsRel, RMSE, delta-1.
- Normal: mean angular error and 5/11.25/22.5 degree accuracy.
- Point map: direct metric MAE/RMSE, support-diagonal normalized RMSE, symmetric Chamfer, and
  F-score at 0.05/0.25/0.50 m. With a 0.25 m grid, 0.25 m is the resolution-matched primary
  threshold; 0.05 m is a strict diagnostic.
- Visibility: IoU and F1.

Each metric must include valid-pixel or valid-point counts and query support coverage. Missing
modalities are not silently treated as zeros.

## Mechanism diagnostics

- Cross-context consistency: compare predictions from two context subsets on identical queries.
- Query permutation: permute target camera order and require output equivariance.
- Intervention specificity: alter density or color fields separately and measure intended versus
  cross-task changes.
- No-bypass audit: fixed renderer parameter count is zero and model forward exposes no target
  labels.
- Uncertainty calibration: bin predicted variance and compare coverage/error when uncertainty is
  enabled.

These diagnostics establish inspectable mechanisms; they do not make arbitrary latent channels
human-interpretable.

## Leakage controls

Do not choose context frames after looking at target RGB/depth. Keep scenes disjoint across train,
validation, and test. Do not use target depth or full-scene manifest bounds to choose the anchor,
state bounds, coordinate normalization, voxel resolution, ray samples, or query support. Report
per-scene and pooled metrics, both support coverages, frame timing, state voxel resolution and
metric spacing, number of ray samples, and peak GPU memory.

## Scene splits and decision order

- Development train: all 365 official Hypersim train scenes.
- Model selection: all 46 official validation scenes.
- Failure analysis only: the 20 official test scenes already evaluated by earlier runs.
- Final report: the remaining 26 official test scenes, evaluated once after freezing the method,
  hyperparameters, training budget, checkpoint-selection rule, and metric code.

The first three execution gates are: tiny real-data smoke, main-resolution BF16 smoke, and
one-window overfit with sustained loss reduction. The expanded main run is not justified merely
because more scenes are available. After the overfit gate, use 4 context views, 2 target views,
stride 2, 150,000 microsteps, and gradient accumulation 2. Validation may select a checkpoint;
neither diagnostic-test nor final-holdout results may feed back into architecture or tuning.
