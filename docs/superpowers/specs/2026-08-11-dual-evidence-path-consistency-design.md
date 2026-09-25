# Dual-Evidence Path-Consistent Scene State Design

## Decision

V6 is an opt-in architecture that preserves the project contract:

```text
calibrated context RGB
-> one target-independent typed 3D state
-> fixed zero-parameter measurement program
-> RGB, depth, normal, point, visibility, and risk diagnostics
```

V6 does not add a target image encoder, a target-conditioned state, a learned renderer,
image-space residual warping, Gaussian primitives, or a task-specific measurement head. The
target camera remains a query. Target labels may be used for supervised losses and post-hoc
audits only; they cannot affect the state, anchor, bounds, evidence, or completion gates.

The implementation is split into two separately evaluated stages:

- V6a separates a deterministic surface-localization proxy from appearance agreement.
- V6b adds evidence-conditioned view-drop stability training and low-overlap path audits.

V5 remains unchanged and is the required matched ablation.

## Evidence motivating the change

The persisted V5 step-5000 report contains one scene and one deterministic training window. It
reports 20.6371 dB PSNR, 0.71965 SSIM, 0.009664 depth AbsRel, 7.046 degree normal error, and
0.99414 F-score at 0.25 m. This is a capacity result, not a generalization result.

V5 logs show mean correspondence evidence near 0.161, unknown probability near 0.839, and a
mean completion gate near 0.855. Most state capacity is therefore completion-dominated. The V5
confidence is computed from cross-view descriptor and RGB agreement at a candidate 3D point. It
does not compare that candidate against alternative depths on the same context rays, so a
textureless or repeated-color region can appear consistent at several incorrect depths.

The accepted appearance oracle found 22.1713 dB for shared doubled-resolution appearance and
only 0.12 dB additional gain from an inadmissible target-specific appearance volume. V6 therefore
keeps one shared appearance state and focuses on surface localization and observation-path
stability.

The V6a depth competition is a fixed local matching/cost-volume cue. It is a necessary state
correction, not a standalone novelty claim. The paper must not describe it as a new alternative to
cost volumes or as a certificate of true occupancy.

## Phase 0: immutable V5 diagnostics

These diagnostics run before V6 code is used for a training decision. They write new artifacts
and never overwrite an existing report or checkpoint.

1. Re-evaluate V5 checkpoints at steps 4000, 4500, and 5000. Each report records checkpoint path,
   SHA-256, resolved config, ray sample count, scene/window IDs, and pooled metrics.
2. Audit the V5 confidence against target depth after inference. Each queried ray is divided into
   surface-near, camera-to-surface free-space, and behind-surface bins. Target depth is an audit
   label only and is never passed to the model. The audit uses 256 fixed uniform samples over the
   configured local ray interval. The surface band is one half of the largest native voxel edge;
   samples closer to the camera are free-space and samples farther away are behind-surface. Rays
   without valid, locally visible target depth are excluded and counted. The untouched validation
   audit uses exactly four windows per scene, selected without labels by taking the four lowest
   `sha256("v6-surface-audit:<scene_id>:<window_start>")` values. V5 and V6a reuse the exact same
   selected records; the eight-scene pilot therefore contains 32 audit windows.
3. Evaluate the same selected checkpoint with 64, 128, and 256 uniform ray samples. A renderer
   change is admitted only if 128 or 256 samples improve PSNR by at least 0.20 dB and do not worsen
   depth AbsRel or normal mean angle by more than 1 percent relative.
4. Re-run the shared and per-view appearance oracle on the selected V5 checkpoint with separate
   provenance reports.
5. Establish a fixed V5 multi-scene pilot on 16 training scenes and 8 untouched validation scenes.
   Scenes are selected without labels by sorting the SHA-256 values of
   `v6-pilot:<split>:<scene_id>` and taking the lowest values. The pilot uses seed 17, 10,000
   micro-steps, the existing four-context/two-target protocol, and no diagnostic-test or final-
   holdout scenes.

If the V5 surface audit already separates surface-near points from both non-surface bins and is
monotonic with held-out geometric error, the necessity of V6a must be reconsidered before a long
run. This prevents renaming an already adequate cost-volume score and presenting it as a new
mechanism.

## V6a typed evidence contract

V6a adds `state_architecture: dual_evidence`. `legacy` and `evidence_residual` construction,
state dictionaries, numerical behavior, and configuration defaults remain unchanged.

The new state evidence payload exposes these native-grid fields:

- `surface_localization_support`: deterministic local-matching support that a target-visible
  surface is localized near the voxel;
- `appearance_confidence`: deterministic cross-view color/descriptor agreement;
- `surface_peakness`: depth-competition component of localization support;
- `localization_unknown_probability = 1 - surface_localization_support`;
- `appearance_unknown_probability = 1 - appearance_confidence`;
- separate localization and appearance completion gates;
- appearance provenance, evidence base density/color, and bounded completion residuals.

The high-resolution `StateAppearance` keeps its existing typed contract, but its confidence is
explicitly appearance confidence. Geometry evidence and density remain on the native metric grid.

V6 uses a new evidence payload type rather than changing the meaning of V5
`StateEvidence.confidence`. This keeps old artifacts interpretable and prevents a V5 field from
being silently relabeled as surface evidence.

## Deterministic dual evidence

Let `a(x)` be the detached V5 cross-view appearance agreement at voxel center `x`. This remains
the V6 appearance confidence:

```text
c_app(x) = a(x)
```

For every valid context view `i`, V6 compares the center candidate against four points displaced
along that view's ray through `x`:

```text
x + k * delta * d_i(x),  k in {-2, -1, 0, 1, 2}
```

`delta` is the smallest native voxel edge length in meters. Native cell edges are computed as
`extent / [W, H, D]`, matching the stored `[D, H, W]` tensor order. Each candidate is scored with the
same detached fixed RGB-pyramid agreement against the other valid context views. A softmax over
the valid depth candidates gives center probability `p_i0`. If reference `i` has `n_i` valid
candidates, peakness is normalized so a flat valid profile is zero and an isolated center peak
approaches one:

```text
peak_i(x) = clamp((n_i * p_i0 - 1) / (n_i - 1), 0, 1)
peak(x) = valid-view mean of peak_i(x)
c_loc(x) = c_app(x) * peak(x)
```

The depth softmax uses an explicit `surface_peak_temperature` configuration value. The initial
V6 configs use 0.05. Evidence calculations run detached in FP32, then cast to the model state
dtype. They contain no learned parameters.

A candidate must remain inside the state AABB and have valid cross-view image support. A reference
view contributes only when the center and at least two competing depth candidates have valid
multi-view support. The valid-view mean divides by contributing references, not total context
views. With no contributing reference, geometry confidence is zero; with fewer than two valid
context views, both confidences are zero. The aggregation is symmetric over context views; the
first context camera defines coordinates but does not receive a privileged evidence weight.

The final decompositions are:

```text
base_density_logits = 4 * c_loc - 2
localization_gate = floor + (1 - floor) * (1 - c_loc)
density_logits = base_density_logits + localization_gate * density_residual

base_color = provenance-weighted context RGB
appearance_gate = floor + (1 - floor) * (1 - c_app)
color = sigmoid(logit(base_color) + appearance_gate * color_logit_residual)
```

The doubled-resolution appearance branch computes `c_app`, provenance, base color, and the
appearance gate at its own resolution. It does not duplicate native-grid surface peakness because
the fixed renderer obtains geometry from the native density field.

## V6b view-drop stability and path audit

V6b training retains the supervised full-context forward path. A second forward path drops one non-anchor
context view. For four context views, the drop index rotates deterministically over indices 1, 2,
and 3; index 0 is always retained because it defines the preselected metric coordinate frame.
Both paths use identical network weights, bounds, target queries, and fixed renderer.

This nested full/drop pair is called `evidence-conditioned view-drop stability`, not independent
observation-path convergence. The stricter mechanism evaluation additionally uses two partially
overlapping three-of-four paths and a five-context audit with paths `{0,1,2}` and `{0,3,4}` that
share only the coordinate anchor. The five-context audit is diagnostic-only and does not replace
the four-context main benchmark.

Let `S_full` and `S_drop` be the two states. The detached overlap masks are:

```text
w_loc = min(c_loc_full, c_loc_drop)
w_app = w_loc * min(c_app_full, c_app_drop)
```

The state loss is a valid-weight-normalized Smooth L1 loss over final density logits under
`w_loc` and final native color under `w_app`. At doubled appearance resolution, `w_loc` is
trilinearly upsampled with `align_corners=False` and multiplied by the minimum of the two
high-resolution appearance-confidence fields. That mask weights the final high-resolution color
loss. A zero-overlap batch contributes exactly zero consistency loss and logs zero overlap; it
cannot produce NaN or be replaced with an unmasked loss.

The primary per-ray overlap mask is computed only from `w_loc` at the renderer's fixed uniform ray
samples, using the maximum sampled support. It is detached and does not depend on either path's
predicted density. A density-weighted overlap is logged only as an auxiliary diagnostic. RGB,
depth, normal, point, and visibility predictions from the two paths are compared on identical
queries under the evidence-only ray weight. Consistency uses Smooth L1 for RGB, depth,
and point, cosine distance for normals, and binary cross entropy for visibility; the existing
measurement weights normalize their contribution. Gradients flow symmetrically into both states.
The full-context measurement loss remains the primary supervised loss. The dropped-path supervised
measurement loss has weight 0.5 so the secondary branch cannot become a consistency-only
teacher/student loop.

The initial V6b loss weights are:

```text
path_state_consistency_weight = 0.05
path_measurement_consistency_weight = 0.10
path_dropped_supervision_weight = 0.50
```

All path weights default to zero in the configuration loader. V6a therefore remains a selectable
ablation. V6b configs use a 500-step warmup and 500-step linear ramp for a 5,000-step overfit run;
the multi-scene pilot uses a 1,000-step warmup and 1,000-step ramp. The 150,000-step schedule is
not created until the pilot gate passes.

Target supervision prevents constant-state collapse. Detached deterministic evidence prevents the
model from lowering confidence to evade consistency. The overlap masks prevent unknown regions
from being forced into the same unsupported completion.

## Risk and uncertainty language

The existing learned variance head is not supervised in the current main configuration. V6 does
not claim it is calibrated uncertainty. Until a held-out calibration audit succeeds, reports and
paper text call it an uncalibrated variance output or omit it.

V6 logs deterministic evidence-risk components separately: geometry unknownness, appearance
unknownness, surface peak ambiguity, context coverage, path overlap, and path disagreement. A
combined uncertainty score is a later calibrated evaluation decision, not a V6 training feature.

## Configuration and compatibility

New model field:

```text
surface_peak_temperature: float = 0.05
```

New training fields, all backward-compatible defaults:

```text
path_state_consistency_weight: float = 0.0
path_measurement_consistency_weight: float = 0.0
path_dropped_supervision_weight: float = 0.0
path_consistency_warmup_steps: int = 0
path_consistency_ramp_steps: int = 0
```

Nonzero path weights require `state_architecture: dual_evidence` and at least three context views.
Learned-head and free-decoder controls receive the same dropped branch, dropped-target supervision,
warmup, training compute, state-consistency term, and output-consistency term. This is required to
separate regularization benefits from the fixed measurement interface.

V6 writes only to `outputs/hypersim_er_v6_*`. It never resumes a V5 checkpoint because the typed
evidence contract and encoder parameters differ. V5 checkpoints remain evaluation-only evidence.

## Diagnostics and logging

Training logs add:

- means and deciles of geometry confidence, appearance confidence, and surface peakness;
- geometry/appearance completion-gate means;
- geometry and appearance high-confidence rewrite energy;
- path voxel-overlap and ray-overlap coverage;
- density, appearance, and per-measurement path disagreement;
- full, dropped, state-consistency, and measurement-consistency loss terms;
- context indices used by each path.

The surface audit writes per-bin counts, confidence histograms, per-scene AUROC/AUPRC, scene-level
bootstrap confidence intervals, Brier score, ECE/reliability bins, and error-by-risk deciles. It
reports surface-versus-free and surface-versus-behind results separately because ray samples are
correlated and occlusion has different meaning. Its label is `target-visible surface-near`, not
global occupancy. Offset count, temperature, and surface band are frozen before untouched
validation audit; evidence-audit scores never select a checkpoint. The four-window SHA-256 rule is
also frozen before either model is audited, so model outputs and target labels cannot choose an
easier validation window.

## Failure handling

- Invalid offset samples are excluded, never padded as agreement.
- Fewer than two valid views yields zero evidence and normalized-or-empty provenance.
- No common evidence yields zero path loss plus an explicit zero-overlap log entry.
- Non-finite evidence, loss, or gradients fail the run before a checkpoint is written.
- Peakness computation is chunked on the native grid to bound GPU memory.
- Target depth or any target label reaching an encoder/evidence API is a test failure.
- Diagnostic-test and final-holdout scene IDs are rejected by V6 pilot configuration checks.

## Acceptance gates

### Static and unit gates

1. All legacy and V5 tests pass unchanged, and a saved V5 state dictionary loads without missing
   or unexpected keys.
2. Dual evidence is finite, detached, in range, view-permutation invariant, and satisfies
   `c_loc <= c_app`.
3. A flat synthetic ray profile has near-zero peakness; a unique correct-depth profile has larger
   geometric confidence than its competing depths.
4. Geometry and appearance reconstruction identities hold exactly within dtype tolerance.
5. Identical states have zero path loss; zero overlap has finite zero loss; confidence receives no
   gradient.
6. The fixed renderer remains parameter-free and no target label enters state construction.
7. Full pytest, Ruff, source compilation, synthetic CLI smoke, CPU forward/backward, and BF16 CUDA
   smoke pass before a real run is recommended.

### Empirical gates

1. V6a must improve scene-macro surface-versus-nonsurface AUROC by at least 0.05 over V5 on the
   frozen validation audit, show lower mean support in both free-space and behind-surface bins,
   retain nontrivial support coverage, and show monotonic held-out error-risk deciles. Otherwise
   the localization proxy is not validated and V6b is blocked.
2. On the one-window gate, V6a must remain within 0.20 dB V5 PSNR and within 5 percent relative on
   depth AbsRel and normal mean angle. This gate checks capacity, not generalization.
3. On the fixed 16/8 pilot, V6b must reduce held-out context-drop state disagreement by at least
   20 percent relative to V6a on the same fixed support. Evidence-overlap coverage may not fall by
   more than 5 percent relative. Normal four-context validation PSNR may not fall by more than
   0.20 dB, and depth AbsRel and normal mean angle may not worsen by more than 2 percent relative.
4. Improvement only in the training consistency loss does not pass. At least one held-out target
   measurement and the context-drop robustness metric must improve.
5. After the pilot passes, architecture, evidence formula, path policy, loss weights, checkpoint
   selection, and metric code are frozen before the 150,000-step main run.

## Paper claim boundary

V6 does not claim the first feed-forward 3D state, the first target-independent reconstruction,
the first cost volume, the first unified depth/NVS model, or calibrated uncertainty.

The claim tested by this design is narrower:

> Calibrated RGB observations construct a target-independent typed state; deterministic local
> matching gates completion, evidence-conditioned view-drop training tests state stability, and a
> fixed measurement interface reads head-free derived measurements without target-specific
> bypasses.

Dual evidence is a necessary state correction, not the standalone novelty. A stronger
observation-path claim is admitted only if the low-overlap five-context audit passes alongside the
view-drop result. The contribution is the combination of auditable completion gating,
evidence-conditioned stability, and the predictor-independent measurement interface. If matched
external baselines remain substantially stronger on every primary metric,
the result is not described as CVPR-ready regardless of mechanism diagnostics.
