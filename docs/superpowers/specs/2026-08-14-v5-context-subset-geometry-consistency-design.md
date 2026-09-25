# V5 Context-Subset Geometry Consistency Design

## Decision

Add one opt-in training mechanism directly to the unchanged V5 evidence-residual model. The
mechanism is named `V5-CSGC` (V5 Context-Subset Geometry Consistency). It is not a V6 revival and
does not use the failed V6a localization supplier.

The preserved project contract is:

```text
context RGB + calibrated cameras
-> V5 evidence-residual typed 3D state
-> doubled-resolution shared appearance
-> fixed zero-parameter renderer
-> target-camera measurements
```

The new path exists only during training. With `context_subset_geometry_weight: 0.0`, it is not
executed. Inference, the V5 state dictionary, state construction, renderer, and target query path
remain unchanged.

## Motivation

V5 constructs geometry as:

```text
base_density_logits = 4 * confidence - 2
completion_gate = floor + (1 - floor) * (1 - confidence)
density_logits = base_density_logits + completion_gate * density_residual
```

The existing rewrite penalty weights residual magnitude by confidence. It therefore constrains
rewriting most strongly where evidence is already strong, while low-confidence regions retain the
largest completion gate. Multi-scene telemetry shows that these completion residuals can approach
their configured bound and vary substantially by scene. V5-CSGC tests whether stabilizing only the
completion residual improves held-out geometry without replacing the V5 evidence mechanism.

The one-window V5 capacity result and the eight-scene result are different protocols. They are not
treated as a same-protocol degradation curve. V5 remains the comparator until the frozen replacement
gate passes.

## Training Paths

For each four-context training batch:

1. The normal four-view V5 forward produces `S4`, all target-camera measurements, and the original
   supervised loss.
2. An independent deterministic generator selects one of the four context indices uniformly. The
   selected RGB image and camera are removed only after the dataset has established the original
   context-local coordinate frame and metric bounds.
3. The remaining three RGB images and cameras pass through the same V5 state encoder with shared
   weights and unchanged bounds, producing `S3`.
4. The three-view completion residual is inserted into the detached four-view evidence base:

```text
z4 = b4 + g4 * r4
z3_to_4 = stopgrad(b4) + stopgrad(g4) * r3
```

This counterfactual isolates completion-residual instability. It does not punish deterministic
evidence changes that are expected when one observation is removed.

## Audited Overlap

Consistency is applied only where the subset path has at least two contributing context views and
where deleting one view does not sharply change deterministic confidence:

```text
shared = count(subset provenance > 1e-6) >= 2
stable = clamp(1 - abs(c4 - c3), 0, 1)
mask = stopgrad(shared * stable)
z_cf = mask * z3_to_4 + (1 - mask) * stopgrad(z4)
```

A zero-overlap batch therefore reproduces the detached teacher density and contributes finite zero
consistency loss. It is never replaced with an unmasked loss.

## Geometry Loss

The existing fixed renderer queries `z_cf` using the original target cameras. Only depth and
visibility enter the new loss:

```text
L_depth = visibility-weighted SmoothL1(depth_cf / voxel_spacing,
                                      stopgrad(depth_4) / voxel_spacing)
L_visibility = SmoothL1(visibility_cf, stopgrad(visibility_4))
L_csgc = L_depth + L_visibility
```

`voxel_spacing` is the mean metric cell edge after mapping stored `[D,H,W]` axes to metric
`[x,y,z]` counts `[W,H,D]`. Depth is weighted by detached four-view visibility. Visibility is also
matched explicitly so the subset path cannot reduce depth loss by making rays transparent.

RGB, high-resolution appearance residuals, point maps, and normals are not consistency targets.
RGB would entangle appearance with the geometry hypothesis. Point is redundant with target rays and
depth. Density-gradient normals are too noisy for the first mechanism test and remain an evaluation
metric.

## Gradient and Leakage Boundary

- The full-context geometry is stopped only inside the new consistency loss. It continues to receive
  the original V5 supervised and regularization gradients.
- The subset path receives the new consistency gradient through `r3`; its confidence, provenance,
  and all full-path evidence fields are detached in the new loss.
- Target RGB, depth, normal, point, visibility, and support never enter either state encoder.
- Target cameras remain renderer queries only.
- No subset-path target supervision is added.

## Configuration and Compatibility

One strict training field is added:

```text
context_subset_geometry_weight: float = 0.0
```

Nonzero use requires exactly four context views, `state_architecture: evidence_residual`, and
`mode: fixed`. The effective weight ramps linearly from zero to its configured value over the
existing optimizer-update `warmup_steps`; if warmup is zero, the configured value applies
immediately. The first frozen proposal is `0.1` with the existing 3,000-update V5 warmup.

No learned module, model field, model parameter, checkpoint model key, or inference branch is added.
When the configured weight is zero, the subset RNG, subset encoder call, counterfactual state, and
consistency telemetry are skipped.

## Required Telemetry

Each enabled JSONL logging step records the dropped and retained indices, full/subset evidence and
gate means, shared-support coverage, stable overlap weight, residual disagreement, rendered depth
drift in voxel units, visibility drift, residual absolute deciles, residual saturation fractions,
and per-scene full/subset evidence, gate, and residual means. Lightweight scalar drift statistics
remain available to the window/EMA tracker, while quantiles and per-scene audits are computed only
when a JSONL record is due. The subset generator state is stored only in enabled-run checkpoints so
an interrupted CSGC run can reproduce its deletion sequence.

Lower residual magnitude or lower training consistency loss alone is not success.

## Acceptance Gates

### Static and software gates

1. All pre-existing tests pass unchanged.
2. A V5 state dictionary loads with no missing or unexpected model keys.
3. Weight zero does not execute the subset path or alter V5 forward/training mathematics.
4. View deletion preserves the original camera matrices, bounds, order of retained views, and
   context-local coordinate frame.
5. Full teacher evidence and predictions receive no gradient from `L_csgc`; subset residuals do.
6. Zero overlap produces finite zero loss.
7. Full pytest, Ruff, source compilation, and an enabled synthetic CPU smoke pass before a real run.

### Empirical gates

1. On the frozen one-window capacity protocol, PSNR may drop by no more than 0.20 dB and depth
   AbsRel and normal mean angle may worsen by no more than 5 percent relative to V5.
2. On held-out context deletion, geometry disagreement must improve by at least 20 percent and
   overlap coverage may fall by no more than 5 percent.
3. On the fixed eight-scene full-context validation, PSNR may drop by no more than 0.20 dB and
   depth AbsRel and normal mean angle may worsen by no more than 2 percent.
4. Replacing V5 additionally requires an absolute held-out geometry improvement. A lower training
   loss or lower path disagreement without better target geometry does not pass.

## Failure Interpretation

- Better consistency with unchanged or worse held-out geometry indicates generic-geometry collapse.
- Better consistency with worse full-context metrics indicates that useful fourth-view evidence is
  being ignored.
- Falling visibility indicates opacity collapse even if depth disagreement improves.
- Falling overlap coverage indicates that the audit region is disappearing rather than stabilizing.
- Approximately two encoder forwards and one additional render per training step are expected; this
  compute increase is not interpreted as a model-capacity change.

No long run, validation tuning, or replacement claim is authorized by this design document.
