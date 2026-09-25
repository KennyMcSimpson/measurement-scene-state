# Evidence-Residual Scene State Design

## Research contract

The new architecture must preserve the project's main claim: calibrated context RGB views are
compressed into one target-independent typed 3D state, and all primary measurements are read by
the existing parameter-free renderer. The target camera is a query only. It must not enter the
state encoder or completion branch.

The architectural change formalizes a bounded residual decomposition:

```
state = context-supported evidence state + completion_gate * completion_residual
```

This is a 3D state residual, not a target-image residual. The model must retain enough auxiliary
fields to audit which state regions are supported by cross-view evidence and which depend on the
learned completion prior.

## Compatibility boundary

- Existing configs default to `state_architecture: legacy` when the field is absent.
- Existing checkpoint parameter names and legacy forward behavior remain unchanged.
- The new architecture uses `state_architecture: evidence_residual` and writes only to new
  `outputs/hypersim_er_*` directories.
- Existing validation and test artifacts remain immutable.
- The 20 already evaluated Hypersim test scenes are diagnostic evidence, not a final holdout.

## Typed evidence contract

`SceneState` gains an optional `StateEvidence` payload with the following fields:

- `confidence`: cross-view correspondence support in `[0, 1]` for every voxel;
- `unknown_probability`: `1 - confidence`, used as the completion gate;
- `provenance`: normalized source-view weights for every voxel;
- `base_density_logits` and `base_color`: the context-supported state before completion;
- `density_residual` and `color_logit_residual`: bounded learned corrections before gating.

The first version does not claim directly observed free space from RGB-only inputs. Free-space
semantics require reliable depth or ray-termination evidence and are deferred rather than
fabricated from camera-frustum coverage.

## Evidence construction

Every voxel center is projected into every context view. The evidence branch samples a
deterministic multi-scale RGB pyramid at those projections, while a separate learned feature
pyramid supplies the 3D completion branch.

For each voxel, evidence confidence combines:

1. the number of context views in which the projection is valid;
2. pairwise cosine agreement of normalized multi-scale RGB descriptors;
3. raw RGB agreement across valid views.

Source provenance is a masked softmax over per-view agreement with the local consensus. Invalid
views receive zero weight. Voxels with fewer than two valid views have zero correspondence
confidence and therefore remain completion-dominated.

Evidence confidence and provenance contain no learned parameters. This prevents the completion
objective from deliberately lowering confidence to gain a larger residual gate or evade the
high-confidence rewrite penalty. Learned sampled features still train normally through the 3D
residual branch.

The context-supported base color is the provenance-weighted sampled RGB. Base density is a
bounded logit transform of correspondence confidence. A 3D residual network receives the fused
observation statistics, evidence confidence, and unknown probability. It predicts bounded density
and color-logit residuals. The residual gate has a small configured floor in supported regions so
the model can correct calibration and Lambertian violations, while unknown regions receive the
full completion capacity. The main bound is 4.0: a 1.5 pilot bound saturated at the 90th percentile
on a one-window diagnostic, while the fixed evidence field itself remained unchanged.

## Interpretability and anti-cheating constraints

- Completion residual heads initialize to zero, so the initial state equals the evidence state.
- Residual magnitude in high-confidence regions is explicitly regularized and logged.
- The renderer remains unchanged and parameter-free.
- The main fixed model does not consume arbitrary latent state features at query time.
- Training logs report mean evidence, unknown probability, completion gate, and residual energy.
- Diagnostics must test provenance normalization, evidence/completion reconstruction identities,
  context-view deletion, density/color intervention specificity, and no target-label bypass.

This architecture is structurally auditable, not fully human-interpretable. The 2D feature pyramid
and 3D residual network remain learned components.

## Capacity change

The evidence-residual main config uses a 48-channel image pyramid, a 48-channel typed state, four
3D refinement blocks, and the existing `[48, 32, 48]` metric grid. This increases capacity without
changing the 0.25 m spatial protocol. Resolution changes remain a separate ablation because they
alter both compute and the geometry metric floor.

## Data expansion protocol

The local official Hypersim metadata contains 365 public train scenes, 46 validation scenes, and
46 test scenes. The current subset contains 60/20/20 scenes. The expanded protocol is:

- train: all 365 official training scenes;
- validation: all 46 official validation scenes;
- diagnostic test: the existing 20 already evaluated official test scenes;
- final holdout: the remaining 26 official test scenes, never evaluated until the architecture,
  checkpoint rule, and hyperparameters are frozen.

Training uses four context views, two target views, and sample stride two, yielding roughly 17,000
windows when about 100 frames are available per scene. The planned main budget is 150,000
micro-steps with gradient accumulation two. A one-window overfit gate and a short multi-scene
validation run must pass before the main run.

Full Hypersim expansion is expected to add substantially less than 100 GiB because the downloader
extracts only selected preview RGB, depth, normal, and camera files. A live disk-space guard must
keep at least 100 GiB free on D before each download batch.

## Evaluation boundary

Architecture and hyperparameters are selected on validation only. The existing diagnostic test
can explain old failures but cannot select the new model. The sealed final holdout is evaluated
once after checkpoint selection.

Hypersim results cannot be compared numerically with published RealEstate10K results. A separate
RealEstate10K/ACID track and matched external baseline reruns are still required for a defensible
NVS comparison.

## Acceptance gates

1. Legacy configs load and legacy model tests remain unchanged.
2. New evidence fields validate shapes, ranges, provenance sums, and device transfer.
3. Evidence-residual forward/backward is finite on CPU and BF16 CUDA smoke.
4. Zero residual reconstructs the base state exactly.
5. High-confidence residual regularization is non-zero only when evidence fields exist.
6. Full unit tests, Ruff, syntax compilation, synthetic CLI smoke, and main-resolution CUDA smoke
   pass before any long training command is recommended.
