# V5 Geometry Oracle Ceilings Design

## Status and decision

This specification defines a diagnostic ladder around the unchanged V5 typed-appearance model.
It does not define a new trainable model, a V6 successor, or a replacement result.

The selected approach is a three-arm diagnostic grouped into two research stages:

1. `R0a-target-fitted`: optimize one shared native-resolution density field against target
   geometry while every V5 parameter and the fixed renderer remain frozen. This is the direct
   capacity ceiling of the current `SceneState` geometry field plus renderer.
2. `R0a-target-projected`: deterministically project target depth labels into
   `p_free/p_surface/p_unknown`, convert the surface probability to physical density, and render
   the resulting state. This calibrates the proposed ray-evidence-to-density mapping.
3. `R0b-context-projected`: apply exactly the same projection and density mapping to context depth,
   seal the state before target-label access, and evaluate it from target cameras. This measures
   whether ideal context-side geometry can drive the existing state and renderer.

The two R0a arms are both required. A fitted-only oracle cannot validate the deterministic mapping
needed by R0b. A projected-only oracle can fail because of the chosen mapping even when the state
and renderer have adequate capacity.

## Alternatives considered

### Direct learned RGB cost volume

This is deferred. A learned cost volume would entangle three possible causes: state/renderer
capacity, context-to-world geometry transport, and RGB geometry inference. A negative result would
repeat the ambiguity of prior branches; a positive result would still not identify the mechanism.

### Target-fitted oracle only

This gives the strongest capacity ceiling but does not test a context-ray evidence representation.
It cannot tell whether a later R0b failure came from the evidence mapping or from context coverage.

### Target-fitted plus target-projected plus context-projected

This is selected. It adds one diagnostic calibration arm but keeps all work outside the production
model. Each transition changes one thing: free density optimization, then a fixed evidence mapping,
then the source of depth evidence.

## Preserved V5 contract

The following files and interfaces are unchanged by R0:

- `MeasurementCompleteSystem.forward(context_rgb, context_cameras, target_cameras, bounds)`;
- `EvidenceResidualStateEncoder` and all V5 trainable parameters;
- the public `SceneState` field contract;
- `FixedMeasurementRenderer` mathematics and parameter count;
- V5 checkpoint keys, authored configs, baseline artifacts, and training/evaluation paths;
- context-local anchor and bounds rules;
- target cameras as measurement queries only in the normal model.

R0 is implemented in standalone diagnostic modules and emits artifacts that cannot be confused with
`mcss.evaluation.v1`. No R0 artifact may be written under
`baselines/v5_typed_appearance_step4500` or an existing training output directory.

## Research questions and stopping logic

The ladder answers the following questions in order:

1. Can the current native-resolution density field and fixed renderer fit target geometry at all?
2. Can a deterministic free/surface/unknown field reproduce that geometry through the renderer?
3. Can the same field built only from context depth recover target geometry where the target surface
   is actually observed by context cameras?

Interpretation is fixed before results are opened:

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| `R0a-target-fitted` fails | Current state resolution, renderer sampling, support, or optimization is limiting | Stop; diagnose state/renderer before adding RGB geometry modules |
| Fitted passes, target-projected fails | The free/surface/unknown fusion or density calibration is wrong | Fix the deterministic mapping; do not start learned evidence |
| Both R0a arms pass, R0b fails on context-observed target surfaces | Context-to-grid transport, coordinate handling, or multi-view fusion is wrong | Fix R0b transport and isolation |
| R0b passes only on context-observed surfaces | Observed geometry is usable; completion remains unsolved | Proceed to aligned/shuffled/constant controls and observed-region protection |
| All three pass while native V5 is weak | Missing context-derived geometry is a supported bottleneck candidate | Proceed to the factorial controls, not directly to a model claim |

No R0 outcome demonstrates RGB-only learnability, validation generalization, or baseline replacement.

## Diagnostic architecture

### New modules

`src/mcss/geometry_oracle.py` owns only diagnostic state construction:

- a target-fitted shared density oracle;
- deterministic ray-depth classification;
- conservative multi-view evidence fusion;
- physical density calibration;
- construction of temporary `SceneState` values with frozen non-geometry fields.

`src/mcss/geometry_oracle_data.py` owns two-stage manifest access:

- pre-seal context RGB, context ray depth, cameras, fixed bounds, and source-path provenance;
- post-seal target cameras and supervision;
- equality checks proving the post-seal window matches the pre-seal context identity.

`src/mcss/geometry_oracle_runner.py` owns selection, checkpoint loading, sealing, rendering,
target-fitted optimization, stratified metrics, immutable artifact writing, and provenance.

`scripts/run_geometry_oracle.py` is a thin entry point. R0 is not added to the ordinary `mcss`
train/evaluate CLI in this phase.

### Public diagnostic records

The production `SceneState` dataclass is not extended. Diagnostic semantics live in separate records:

```python
@dataclass(frozen=True)
class RayGeometryEvidence:
    p_free: Tensor       # [B, 1, D, H, W]
    p_surface: Tensor    # [B, 1, D, H, W]
    p_unknown: Tensor    # [B, 1, D, H, W]
    conflict: Tensor     # bool [B, 1, D, H, W]
    observed_views: Tensor


@dataclass(frozen=True)
class ContextDepthOracleInput:
    ray_depth_m: Tensor       # [B, V_context, 1, H, W]
    context_cameras: Cameras
    bounds: Tensor            # [B, 2, 3]
```

`ContextDepthOracleInput` contains no target camera, target label, manifest handle, `SceneBatch`, or
arbitrary keyword mapping. `build_context_depth_oracle` accepts this record and a detached reference
state only. The target-label wrapper is separate and explicitly marked inadmissible.

## Two-stage access and leakage isolation

R0 does not add `context_depth` to `SceneExample` or `SceneBatch`. Those objects eagerly contain
target supervision, which is convenient for training but too weak for a privileged-input audit.

For every explicit window index, the runner follows this order:

1. Resolve manifest and frame identities without opening frame payloads.
2. Open only context RGB, context depth, context intrinsics, and context poses.
3. Apply the existing image resizing, intrinsic scaling, first-context anchor transform, and fixed
   `context_local_metric` bounds.
4. Run the frozen V5 state encoder from context RGB.
5. Build `R0b-context-projected` using only `ContextDepthOracleInput`.
6. Hash the context inputs, native state, R0b evidence, R0b state, config, and checkpoint.
7. Atomically write a per-window seal record and append the pre-seal access log.
8. Only after the seal exists, open target cameras and target supervision.
9. Verify the post-seal example has identical context RGB, cameras, bounds, scene ID, and frame
   identity to the sealed context window.
10. Build the two target-label R0a arms and score all arms.

Target cameras and labels cannot choose bounds, anchor, resolution, surface-band policy, density
calibration, window selection, or the context-observed threshold policy. The fixed half-cell policy
is evaluated at each post-seal target point, so its numerical value may vary with the corresponding
context-ray direction; target data never tunes its scale or formula.

Tests poison all target file reads during steps 1-7. Mutating target labels must leave the sealed R0b
state hash unchanged; mutating context depth must change it.

## Coordinate and depth semantics

Cameras use OpenCV camera-to-world matrices with normalized rays. Manifest depth in this project is
metric Euclidean distance along the normalized ray, not OpenCV camera-z depth.

For native resolution `[D, H, W]`, voxel centers are:

```text
x[d,h,w] = bounds_min
           + ((w + 0.5) / W, (h + 0.5) / H, (d + 0.5) / D)
           * (bounds_max - bounds_min)
```

Tensor axes remain `[D, H, W]`; world coordinates remain `[x, y, z]`, with x mapped to W, y to H,
and z to D.

Depth is resized with nearest-neighbor interpolation. Intrinsics are scaled by the same width and
height ratios used by the existing manifest loader. Rigid anchor rebasing changes ray origins and
directions but not metric ray length.

## Ray evidence semantics

For voxel center `x_i` and observation view `v`:

1. Project `x_i` into the image and sample the nearest depth label `d_vi`.
2. Let `o_v` be the camera origin and `t_vi = ||x_i - o_v||_2`.
3. Let the ray direction be `r_vi = (x_i - o_v) / t_vi`.
4. Let voxel edge lengths in world x/y/z be `delta_x`, `delta_y`, `delta_z`.
5. Define the half-cell ray band:

```text
h_vi = 0.5 * min(delta_axis / abs(r_vi_axis))
```

Axes with `abs(r_vi_axis) <= eps` are excluded from the minimum. The band is therefore derived only
from fixed bounds, resolution, and camera geometry.

A view contributes mutually exclusive hard evidence only when projection is in-frame and sampled
depth is finite and positive. The target-label wrapper additionally intersects validity with target
visibility and target support when those fields are present. The context wrapper has no target-side
mask:

```text
free_vi    = t_vi < d_vi - h_vi
surface_vi = abs(t_vi - d_vi) <= h_vi
behind_vi  = t_vi > d_vi + h_vi
```

Invalid and behind-surface observations provide no free-space or surface claim. Space behind the
first depth return remains unknown.

Across views:

```text
any_free    = OR_v free_vi
any_surface = OR_v surface_vi
conflict    = any_free AND any_surface

p_free    = any_free AND NOT any_surface
p_surface = any_surface AND NOT any_free
p_unknown = 1 - p_free - p_surface
```

Conflicts become unknown. The three probabilities are finite, non-negative, mutually exclusive,
and sum to one. Context-view permutation must leave the fused result bit-identical.

## Evidence-to-density calibration

The fixed renderer consumes an extinction coefficient `sigma = softplus(density_logits)`. It does
not consume free or unknown probabilities directly. R0 uses the same fixed mapping for
`R0a-target-projected` and `R0b-context-projected`:

```text
cell_length     = min(delta_x, delta_y, delta_z)
empty_opacity   = 1e-6
surface_opacity = 0.99

opacity = empty_opacity
          + (surface_opacity - empty_opacity) * p_surface
sigma   = -log1p(-opacity) / cell_length
density_logits = softplus_inverse(sigma)
```

The stable finite inverse is:

```text
softplus_inverse(sigma) = sigma + log(-expm1(-sigma))
```

The nonzero empty opacity avoids `-inf`, which `SceneState` correctly rejects. Free and unknown
remain distinct in `RayGeometryEvidence` but both map to effectively empty density in the current
renderer. This is intentional: R0 does not change the renderer contract.

## R0a target-fitted ceiling

`R0a-target-fitted` owns one trainable tensor: a clone of the native V5 `density_logits` with the
same `[B, 1, D, H, W]` shape and bounds. It creates temporary states with frozen detached V5 color,
appearance, log variance, and bounds. It never optimizes a per-target-view state.

The fixed renderer produces depth, normal, point, and visibility. The oracle objective reuses the
existing geometry losses and weights:

```text
L_fit = 1.00 * L_depth_relative
      + 0.50 * L_normal_cosine
      + 0.25 * L_point_L1
      + 0.50 * L_visibility_BCE
```

Only requested and available labels contribute. RGB and uncertainty are excluded. Adam optimizes
only oracle density; CLI arguments explicitly set oracle steps and learning rate. The runner stores
initial, best, final, and last-20-percent loss traces, and restores the best density before scoring.

`model_training_updates` is always zero. `oracle_optimizer_steps` is reported separately. Before
and after hashes must prove that the checkpoint and frozen model parameters did not change.

## Temporary state fields

Each oracle state contains:

- oracle density logits;
- detached V5 native color;
- detached V5 log variance;
- exact detached V5 bounds;
- detached V5 typed appearance when present, used only for descriptive RGB rendering;
- no features and no native/dual/transport evidence payload.

Primary R0 gates use geometry only. RGB is optional descriptive output and cannot decide a geometry
gate. Uncertainty is not scored because neither R0 geometry arm defines a calibrated uncertainty
oracle.

## Evaluation masks

Masks are frozen before prediction metrics are opened and never depend on a prediction.

Every arm is reported on:

1. `local_support_all`: the existing camera-and-bounds target support.
2. `context_observed_surface`: supported visible target pixels whose 3D target point reprojects into
   at least one context view with a finite positive context depth and whose context-camera ray range
   differs from that depth by no more than the same half-cell ray band.
3. `target_visible_not_context_observed`: supported visible target pixels outside mask 2.
4. `outside_local_support`: count and coverage only; clipped labels are not scored.

The context-observed mask uses target depth only after the R0b state seal and only for evaluation
stratification. Reports declare this explicitly. Coverage is a primary endpoint for every stratum.

Metrics are computed with the existing metric functions and include at least:

- depth AbsRel, RMSE, and delta-1;
- normal mean angle and threshold accuracies;
- point MAE, RMSE, Chamfer, and F-score at 0.25 m;
- visibility IoU and F1;
- valid count and coverage for each metric/mask.

Additional renderer diagnostics are mean opacity, low-opacity fraction, conditional depth
`depth / visibility`, conflict fraction, surface-voxel fraction, and observed-view-count summaries.
The native renderer sample count is primary. The same sealed state is also rendered descriptively at
2x and 4x `n_samples`; improvement there identifies midpoint-sampling pressure but cannot replace the
native-renderer result.

## Predeclared decision gates

Let `s` be the mean native voxel edge length in metres.

### R0a target-fitted capacity gate

The gate is positive only when all conditions hold on `local_support_all`:

- depth RMSE is at most `s`;
- point MAE is at most `s`;
- normal mean angle is at most 20 degrees;
- visibility F1 is at least 0.95;
- the best fit improves native V5 depth RMSE and at least one of point MAE or normal angle;
- the last 20 percent of optimization improves total loss by less than 2 percent.

Failure with a still-improving tail is `INCONCLUSIVE_OPTIMIZATION`, not a capacity rejection.

### R0a target-projected mapping gate

The gate is positive only when:

- depth RMSE is at most `1.5 * s`;
- point MAE is at most `1.5 * s`;
- visibility F1 is at least 0.90;
- conflict fraction is reported and does not exceed 5 percent of voxels receiving any evidence.

If target-fitted passes and target-projected fails, only the mapping is rejected.

### R0b context-projected evidence gate

R0b is first assessed only on `context_observed_surface`. It is `INCONCLUSIVE_COVERAGE` when that
mask covers less than 10 percent of supported visible target pixels.

Otherwise, calculate native-to-ceiling gap closure for lower-is-better metric `m`:

```text
gap_closure(m) = (m_native - m_context) / (m_native - m_target_fitted)
```

The denominator must be positive; otherwise that metric is inconclusive. R0b reaches the next-stage
candidate gate only when median gap closure is at least 0.50 for depth RMSE and point MAE and
visibility F1 is no more than 0.02 below native V5 on the same mask.

This status authorizes shuffled/constant controls. It is not evidence for a learned RGB method.

## Artifacts and provenance

The runner requires an explicit fresh output directory and one or more explicit window indices. It
has no implicit `dataset[0]` and no overwrite flag.

It writes:

- `r0_manifest.json`: schema, arguments, hashes, component status, and fixed policies;
- `seals/window_<index>.json`: pre-target-access state/input hashes;
- `r0_windows.jsonl`: one immutable result record per selected window;
- `r0_report.json`: aggregate metrics and gate statuses;
- `access_log.json`: ordered context and target file access with pre/post-seal phase;
- `artifact_hashes.json`: hashes for every completed diagnostic artifact except the hash manifest
  itself.

Required top-level fields include:

```text
schema_version: mcss.geometry_oracle.v1
diagnostic_only: true
result_class: non_comparable_geometry_diagnostic
comparison_eligible_as_v5_training_result: false
model_training_updates: 0
target_labels_used_by_normal_model: false
target_labels_used_by_r0a: true
target_labels_used_by_r0b_state: false
target_labels_used_by_r0b_evaluation: true
```

The report also records absolute config/checkpoint paths and hashes, checkpoint step, model-config
compatibility, renderer settings, grid/bounds/depth conventions, selected scenes/frames/windows,
oracle hyperparameters, optimized parameter counts, access-order checks, metric definitions, and
aggregation rules.

Only model-architecture fields must match the checkpoint embedded config. Dataset roots, selected
windows, and output directories may intentionally differ for evaluation. The checkpoint is hashed
again after the run and mutation fails closed.

## TDD and verification requirements

Implementation follows RED-GREEN-REFACTOR in these slices:

1. Analytic ray classification and conflict-to-unknown fusion.
2. Finite probability normalization, batch/device/dtype behavior, and view-permutation invariance.
3. Physical opacity-to-density calibration and temporary `SceneState` construction.
4. Target-fitted density ownership: only oracle density receives gradients; model, renderer, and
   copied typed fields remain frozen.
5. Two-stage manifest access, seal-before-target-read, target mutation noninterference, and context
   mutation sensitivity.
6. Context-observed evaluation mask and fixed coverage accounting.
7. Immutable runner artifacts, exact provenance, no overwrite, and checkpoint/model hash stability.
8. Frozen V5 checkpoint strict loading and unchanged target-label-free main forward signature.

Focused verification commands are:

```powershell
uv run --extra dev pytest tests/test_geometry_oracle.py -q
uv run --extra dev pytest tests/test_geometry_oracle_data.py -q
uv run --extra dev pytest tests/test_geometry_oracle_runner.py -q
uv run --extra dev pytest tests/test_geometry_oracle.py tests/test_geometry_oracle_data.py tests/test_geometry_oracle_runner.py tests/test_geometry.py tests/test_measurements.py tests/test_model.py tests/test_v5_checkpoint_compatibility.py -q
uv run --extra dev ruff check src/mcss/geometry_oracle.py src/mcss/geometry_oracle_data.py src/mcss/geometry_oracle_runner.py scripts/run_geometry_oracle.py tests/test_geometry_oracle.py tests/test_geometry_oracle_data.py tests/test_geometry_oracle_runner.py
uv run python -m compileall -q src scripts tests
uv run --extra dev pytest -q
```

Static tests prove software contracts and leakage isolation. They do not validate scientific
performance. Scientific gates require fresh R0 artifacts on the predeclared windows.

## Scope after R0

No learned RGB geometry module or hard protection is implemented in this specification.

If the R0b candidate gate is reached, the next specification must implement a fixed factorial over
`native/oracle/shuffled/constant` evidence and `protection off/on`. Shuffles must include both
depth-to-camera derangement and cross-scene/camera mismatch. Constant evidence is a severity
baseline, not a permutation test. Hard protection may use only information available to the arm
being tested; context-depth masks remain privileged controls and cannot enter the RGB-only arm.

Only after that factorial identifies an aligned-evidence effect may a new specification introduce
RGB-only learned context-ray evidence.

## Repository constraint

`<workspace-root>` is not a Git repository. This specification can be written
and verified in place but cannot be committed there. No commit claim is permitted unless the source
root is later placed under version control by an explicit user decision.
