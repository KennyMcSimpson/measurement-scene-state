# V7 MapAnything-Certified Observation Audit Design

## Decision and scope

V7 replaces only the sealed LoFTR observation inlet. It preserves the project backbone:

```text
context RGB + calibrated OpenCV cameras
-> target-independent typed metric state
-> fixed zero-parameter measurement program
-> RGB / depth / normal / point / visibility / risk
```

The candidate inlet is:

```text
context RGB + known intrinsics/cam2world
-> frozen MapAnything metric posed-MVS inference
-> project-owned known-camera consistency certification
-> conditional free/surface/unknown/conflict/provenance/risk observations
```

This stage is a supplier qualification audit only. It does not modify the production typed state,
completion network, training loop, renderer, V5/V6 results, diagnostic-test partition, or final
holdout. Training remains blocked unless every frozen audit gate passes.

## Claim boundary

MapAnything, metric depth prediction, camera-conditioned MVS, reprojection, depth consistency,
free-space intervals, occupancy semantics, and confidence are not project contributions. The
supplier is an external frozen prior. A successful audit would establish only that it is a viable
source of observations for a later project-owned write-protection experiment.

The provisional paper contribution remains the combined contract: an external predictor may
propose geometry; known-camera checks decide write authority; certified geometry is immutable;
completion may write only unknown regions; one fixed renderer reads all measurements. This audit
cannot establish that contribution or CVPR readiness by itself.

## Frozen supplier identity

- Code repository: `facebookresearch/map-anything`
- Code revision: `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`
- DINOv2 code repository: `facebookresearch/dinov2`
- DINOv2 code revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- DINOv2 tracked-source SHA-256:
  `ef177de4d1146157c59515f3fcc931767ff489d80309f722fdb3fd10519d0c94`
- Model repository: `facebook/map-anything`
- Model revision: `a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa`
- Model license: CC BY-NC 4.0; code license: Apache 2.0
- Inference: frozen, evaluation mode, `torch.inference_mode`, BF16 autocast,
  memory-efficient inference, minibatch size 1
- Inputs: context RGB, known intrinsics, known OpenCV cam2world, and
  `is_metric_scale=True`
- Forbidden inputs: every context depth/normal/point label and every target RGB/depth/normal/
  point/visibility/camera

MapAnything's pinned `_from_pretrained` path forces the UniCeption DINOv2 encoder to use
`torch_hub_pretrained=False`. DINOv2 therefore supplies architecture code only: there is no
separate DINOv2 checkpoint in this audit, and all encoder parameters are restored from the pinned
MapAnything `model.safetensors` (SHA-256
`981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0`). The adapter verifies a
clean local DINOv2 checkout at the frozen revision and source hash, reroutes UniCeption's
hard-coded hub call to that local checkout, and rejects remote hub repositories, independent
DINOv2 pretrained weights, and URL weight downloads while the model is constructed.

Frame IDs are sorted canonically before inference. Model-predicted camera poses, intrinsics, and
world points are ignored. The project reconstructs world points from `depth_along_ray` and the
known cameras passed through the official preprocessing path.

## Atomic cache and label isolation

Each frozen window produces two structured, overwrite-refusing caches:

1. supplier cache: JSON metadata plus NPZ arrays for depth, raw confidence, mask, processed known
   intrinsics, and known cam2world;
2. certificate cache: JSON metadata plus NPZ arrays for typed intervals, support/conflict counts,
   provenance, risk, and certified masks.

Both are built in a temporary sibling directory, hashed, and atomically renamed. Metadata records
the model and code revisions, frame IDs, input RGB/camera hashes, exact inference configuration,
array schema, and artifact hash. Pickle is forbidden. Existing cache destinations are never
overwritten.

Only after both caches are sealed may the evaluator open context depth. Access timestamps and
paths are recorded. The evaluator has no path back into supplier inference or certification.

## Known-camera certification

For every sampled source ray, the project reconstructs a world point from predicted along-ray
depth. It projects that point into each other known camera and samples the other predicted depth.
A target view supports the source observation only when all of these frozen conditions hold:

- both supplier masks are valid and all values are finite;
- source and target depths are in `[0.10 m, 20.0 m]`;
- the projected point is in front of and inside the target camera;
- along-ray disagreement is at most `0.10 m + 0.05 * projected_distance`;
- the target reconstruction projects back within `2.0 px` of the source pixel.

If the target prediction is materially nearer, it is recorded as an occlusion. Other inconsistent
visible predictions are conflicts. Certification requires at least one supporting view and zero
conflicting views. Samples are evaluated on a fixed four-pixel grid. These values are frozen
before context depth is opened and are not tuned after audit results are known.

Every certified sample carries:

- a conditional free prefix ending `0.05 m` before the predicted surface;
- a surface interval of `+/- 0.05 m`;
- behind-surface unknown status;
- support, occlusion, and conflict counts;
- a per-view provenance bitset;
- a fixed risk score combining support deficit, normalized consistency residuals, conflict rate,
  and raw supplier confidence.

This is a conditional ray certificate, not direct free-space measurement. Its validity depends on
the supplier prediction, known camera calibration, visibility assumptions, and frozen consistency
checks.

## Frozen windows and endpoints

The audit reuses the exact 32 scene-balanced, seed-17 Hypersim development-train windows in
`outputs/geometry_certificate_audit_v1/frozen_windows.json`. Diagnostic-test and the 26-scene final
holdout are forbidden. Four context views are used. No favorable-window reselection is allowed.

Primary endpoints are macro certified grid coverage, nonzero-scene fraction, and source-ray 3D
error against context depth opened after sealing. Supporting endpoints are raw-valid coverage,
support/conflict/occlusion distributions, same-coverage raw-confidence error, risk-error Spearman
correlation, camera/correspondence shuffles, and canonical view-order invariance.

## Frozen decision rule

The branch receives `GO_TO_STATE_INTEGRATION` only if all gates pass:

- exactly 32 frozen windows complete;
- macro certified coverage is at least `15%`;
- at least `80%` of windows have nonzero certified observations;
- median source-ray 3D error is at most `0.25 m`;
- P90 source-ray 3D error is at most `0.50 m`;
- certified observations beat raw MapAnything confidence at identical evaluated coverage;
- risk-error Spearman correlation is strictly positive;
- replaying a reversed input order through the canonical frame-ID ordering path changes aligned
  depth by at most `1e-4 m` (a determinism/order-normalization check, not a claim of native model
  permutation equivariance);
- a frozen camera/correspondence shuffle either halves certified coverage or raises median error by
  both `1.5x` and `0.05 m`.

Missing endpoints yield `INCONCLUSIVE`. Any completed failed gate yields
`STOP_MAPANYTHING_SUPPLIER`. A STOP cannot be rescued by threshold adjustment, scene selection,
longer training, or a stronger completion model. A different supplier or threshold set is a new
pre-registered branch.

## Provenance and limitations

The final report must include code/model hashes, model-card license, disk usage, GPU/runtime,
pretraining-data disclosure available from the official paper/model card, every accessed label
path, and artifact hashes. The official repository does not list Hypersim among its 13 training
dataset integrations, but absence of direct overlap is not claimed without a complete training
data audit. Formal Research Institute and Evaluation Council status must be reported honestly.
