# Geometry-Certified Write-Protected State: Label-Blind Audit Design

## Decision and scope

This is a new research branch after the sealed V6a.1 mechanism failure. It is not V6a.2 and does
not modify V5, V6a, V6a.1, the typed state, or the fixed renderer. The first executable stage is a
zero-training audit. Training is blocked until this audit establishes that context RGB and
calibrated context cameras can produce useful, independently checkable 3D observations.

The preserved project backbone is:

```text
multiple calibrated context RGB observations
-> target-independent typed metric 3D state
-> fixed zero-parameter measurement program
-> RGB / depth / normal / point / visibility / risk
```

The candidate extension is:

```text
context-only correspondence proposals
-> reciprocal + epipolar + cycle + cheirality + reprojection certification
-> conditional ray certificates
-> certified free prefix / surface interval / behind-surface unknown / conflict / provenance
-> unknown-only completion
-> unchanged fixed measurement program
```

This stage implements and tests only the certificate builder and its post-hoc evaluator. It does
not yet write certificates into the production state or train a completion network.

## Claim boundary

Matching, triangulation, epipolar filtering, occupancy, free/occupied/unknown semantics, TSDFs,
and preserving measurements are not claimed as individual contributions. G2SR already occupies a
direct learned-proposal plus geometric-triangulation route, while MatchNeRF, EpiS, MVSplat,
MVSGaussian, eFreeSplat, VGGT, MapAnything, NeuralRecon, TransformerFusion, TSDF/OctoMap, and scene
completion occupy important neighboring components.

The only provisional contribution is the combined, counterfactually testable contract:

- learning or image similarity may propose an observation but cannot certify it;
- fixed geometry decides what may enter the measured state;
- completion may write only unknown regions and cannot overwrite a certificate;
- one fixed renderer reads every measurement;
- every output remains attributable to measured or completed evidence.

The NeuralRecon/TransformerFusion/TSDF/scene-completion equivalence check remains incomplete. The
arXiv interface returned HTTP 429 during the pre-implementation check. No CVPR novelty claim is
authorized by this design or by a passing audit.

## Frozen audit protocol

The audit uses only prepared Hypersim development-train scenes. Final-holdout and diagnostic-test
partitions are excluded. Windows are selected deterministically from scene IDs and the fixed seed
`17`. Four consecutive context frames are used; no target frame is required for certificate
construction.

The proposer is a frozen Kornia LoFTR model with ScanNet `indoor_new` weights. It is context-only,
receives grayscale image pairs, is never fine-tuned, and is not a claimed contribution. Its native
dual-softmax confidence threshold is fixed at `0.2`. Geometry, rather than LoFTR confidence,
determines whether a proposal may be certified. The exact weights file and SHA-256 are recorded by
the runner.

Certification is cumulative:

1. `raw`: LoFTR proposes a finite pair with native confidence at least `0.2`;
2. `reciprocal`: the reverse pair set returns within `1.5 px` in both images;
3. `cycle`: source -> matched -> deterministic third view -> source closes within `2.0 px`;
4. `geometric`: the proposal lies within `1.5 px` of the calibrated epipolar line, DLT
   triangulation has positive depth in both source views, triangulation angle is at least
   `1.0 degree`, and source-pair reprojection error is at most `1.5 px`;
5. `certified`: the triangulated point reprojects into the third context view and agrees with the
   cycle proposal within `2.0 px`.

All thresholds are frozen before reading any depth labels. No threshold sweep is allowed after
post-hoc scoring.

## Label isolation

`build_certificates(context_rgb, context_cameras, config)` is the only certificate-construction
entry point. Its signature cannot receive target RGB, depth, normal, point, visibility, or target
cameras. `evaluate_certificates(certificates, context_depth, context_cameras, ...)` is a separate
post-hoc function. Evaluation labels cannot alter a certificate, threshold, selected window, or
decision rule.

Prepared context-frame depth is loaded only after the immutable certificate JSON has been written.
Depth provides post-hoc 3D correctness for audit purposes and is never presented as an inference
input or a model result.

## Endpoints and controls

Primary endpoint:

- certified source-pixel coverage: certified source samples divided by fixed grid samples, macro
  averaged over windows.

Supportive endpoints:

- counts and retention at raw, reciprocal, cycle, geometric, and certified stages;
- median and p90 source-pair and third-view reprojection error;
- median triangulation angle;
- post-hoc 3D error against source context depth;
- matched-coverage comparison between geometric certificates and ordinary LoFTR confidence;
- occupied native-grid coverage at `48 x 32 x 48` within the existing local metric bounds;
- certificate invariance under context-view permutation;
- shuffled-camera and shuffled-correspondence negative controls.

## Frozen decision rule

The branch receives `GO_TO_STATE_INTEGRATION` only if all conditions hold on at least 32 frozen
development-train windows:

- macro certified source coverage is at least `15%`;
- at least `80%` of windows have nonzero certified observations;
- median post-hoc 3D error is at most one native voxel (`0.25 m`);
- p90 post-hoc 3D error is at most two native voxels (`0.50 m`);
- certification beats ordinary descriptor confidence at matched coverage on median 3D error;
- the predeclared certificate risk has strictly positive Spearman correlation with post-hoc 3D
  error;
- view-permutation geometry differs by at most numerical tolerance;
- shuffled controls materially degrade geometry correctness.

Any failed primary condition yields `STOP_CERTIFICATE_FRONTEND`. An incomplete run yields
`INCONCLUSIVE`. A stopped frontend is not rescued by training, threshold tuning, a stronger
completion network, or reading validation/test labels. Changing the proposer or its weights is a
new, predeclared audit, not a reinterpretation of this frozen result.

## Super Research status

The earlier medium-intensity preparation used three non-formal consultants because the formal
Research Institute route and requested Luna route were unavailable. The current code stage does
not manufacture missing Institute records. Research Institute roles not invoked are reported as
not invoked. Council not convened. The Generation 10 terminal-report document is a design, while
the active ecosystem remains Generation 9.
