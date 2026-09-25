# Geometry Certificate Frontend Failure Seal

## Decision

The frozen 32-window label-blind audit is sealed as `STOP_CERTIFICATE_FRONTEND`. Do not connect
this exact LoFTR-indoor geometry-certificate frontend to the typed state or start model training.
Do not tune thresholds, select favorable scenes, change weights, or use completion to conceal the
failed observation contract.

This seal does not reject the original project backbone or write protection itself. It rejects the
current A-side measurement inlet:

```text
LoFTR indoor proposals
-> reciprocal + 3-cycle + epipolar + DLT + cheirality + third-view reprojection
-> conditional geometry certificates
```

## Frozen evidence

The audit used seed 17, 32 hash-selected development-train scenes, four context views, 128x160
images, no target frame, and the predeclared thresholds in
`configs/hypersim_geometry_certificate_audit.yaml`. Every certificate JSON was serialized before
the corresponding context depth was opened for post-hoc scoring. No diagnostic-test or final
holdout file was accessed.

| Gate | Result | Threshold | Decision |
|---|---:|---:|---|
| Certified source-grid coverage | 2.5952% | >= 15% | FAIL |
| Nonzero-window fraction | 81.25% | >= 80% | PASS |
| Median 3D error | 1.5179 m | <= 0.25 m | FAIL |
| P90 3D error | 12.2370 m | <= 0.50 m | FAIL |
| Matched-confidence median | 3.8603 m | certificate must be lower | PASS |
| View-permutation point delta | 0.01105 m | <= 0.0001 m | FAIL |
| Shuffled-match median | 13.5633 m | materially worse | PASS |
| Risk-error Spearman | 0.1945 | > 0 | PASS |

The stage flow was `20,804 raw -> 12,668 reciprocal -> 5,318 cycle -> 3,044 geometric -> 1,420
certified`. Only 1,417 certificates had valid post-hoc depth in all required views. Native-grid
occupied coverage was 0.0241% on average.

## Interpretation

The controls show the code is responsive rather than completely broken: camera and correspondence
shuffles degrade error, geometry certification beats LoFTR confidence at matched count, and the
predeclared risk is positively associated with error. However, reciprocal and cycle consistency,
small pixel reprojection error, and cheirality still admit correspondences on repeated structures,
occlusion boundaries, or incorrect surfaces. Pixel-space closure is therefore insufficient to
grant immutable measurement-write authority.

The failure is decisive for state integration because the certificate coverage is about 5.8 times
below the floor, median error is about 6.1 times above one voxel, and p90 error is about 24.5 times
above two voxels. A completion model would receive sparse and geometrically unsafe protected
measurements.

## What is ruled out

- training this exact frontend;
- weakening the 15% coverage or 0.25/0.50 m geometry gates;
- selecting scenes or thresholds after reading depth;
- treating third-view cycle/reprojection consistency as a geometry certificate by itself;
- claiming the A+B method, external-baseline performance, or CVPR readiness from this audit.

## What remains open

- the original context-to-typed-state-to-fixed-renderer backbone;
- write protection as a contract, provided a future observation source is separately justified;
- an explicit dense posterior baseline as a performance reference, not a novelty claim;
- a new research direction discussed with the user before code changes.

Any future proposer, pretrained weights, scene sampling, depth posterior, or observation source is
a new predeclared branch. It cannot be reported as tuning or continuation of this sealed audit.

## Authoritative artifacts

- `outputs/geometry_certificate_audit_v1/audit_result.json`
- `outputs/geometry_certificate_audit_v1/REPORT.md`
- `outputs/geometry_certificate_audit_v1/frozen_windows.json`
- `outputs/geometry_certificate_audit_v1/window_metrics.jsonl`
- `outputs/geometry_certificate_audit_v1/access_log.json`
- `outputs/geometry_certificate_audit_v1/artifact_hashes.json`

The formal Research Institute was not invoked and the Evaluation Council was not convened. This is
a machine-executed project gate, not Institute or Council PASS.

