# V6a.1 Transport Failure Seal

## Decision

V6a.1 is sealed as `FAIL_CAPACITY_DEPTH_NORMAL`. Stop the complete V6
localization-gating branch: no longer training, no 32-window surface audit, no V6b, no 16/8
pilot, and no external-baseline or CVPR-ready claim.

This is a mechanism failure, not a runtime or implementation failure. The implementation remains
as a versioned negative ablation.

## Frozen capacity gate

The candidate and reference use the same seed, Hypersim scene/window, 4 context views, 2 target
views, 128x160 images, 48x32x48 metric state, and 64 fixed-renderer samples.

| Metric | V5 step 4500 | V6a step 5000 | V6a.1 step 5000 | V6a.1 vs V5 | Gate |
|---|---:|---:|---:|---:|---|
| RGB PSNR | 21.065804 | 21.018525 | 20.975241 | -0.090563 dB | PASS |
| Depth AbsRel | 0.00889812 | 0.00959913 | 0.00988325 | +11.0712% | FAIL |
| Normal mean angle | 6.499368 | 6.186040 | 6.954568 | +7.0038% | FAIL |

V6a.1 also worsens over the already failed V6a by 2.9599% AbsRel and 12.4236% normal angle.

Authoritative artifacts:

- `outputs/hypersim_er_v6a1_transport_capacity_gate_step5000/capacity_gate_report.json`
- `outputs/hypersim_er_v6a1_transport_overfit_eval_step5000/evaluation_report.json`
- `outputs/hypersim_er_v5_appearance2x_overfit_eval_step4500_reference/evaluation_report.json`
- V6a.1 checkpoint SHA-256:
  `ff9f796cdc87252205dcedd29a30744031d996db0f33ba6b85357cbb014e16fb`

## Mechanism diagnosis

Transport fixes where off-center evidence is written, but it does not turn the fixed local RGB
competition into a dense or reliable surface likelihood.

- Only 6.636% of source profiles contribute after requiring two other observers and at least three
  valid candidates.
- Transport coverage is nonzero in only 12.515% of native voxels; its median is zero.
- Surface localization support has mean 0.001950, median zero, and p90 0.002506.
- The localization completion gate has mean 0.998245, so the learned residual remains almost fully
  open throughout the grid.
- The final density residual has mean absolute magnitude 2.25, while localization-weighted rewrite
  energy is only 0.004991.
- Candidate competition has median normalized entropy 0.9979. Center fractional argmax is 7.49%
  versus a 21.09% chance baseline.
- 78.0% of unique winners lie on the -2 or +2 search boundary. Transport therefore moves a signal
  dominated by boundary, visibility, occlusion, or appearance-gradient effects rather than a
  centered local surface likelihood.

The resulting density base is almost uniformly empty (`4 * c_loc - 2` has mean approximately
-1.9922), and the predictor must reconstruct geometry through a large residual. This can retain
RGB while shifting integrated surface depth and destabilizing normals.

## What is ruled out

The following are not justified follow-ups:

- more V6a.1 steps;
- changing temperature, offset range, or observer minimum after seeing target metrics;
- another mapping from the same local RGB competition into the density base;
- weakening the frozen 5% thresholds;
- selecting an intermediate checkpoint;
- claiming the V6 evidence path improves the V5 typed state.

Two opposite placement interventions have now failed: V6a leaves the same evidence at the source
center, while V6a.1 transports it to the winning candidate. The failure is no longer explained by
placement alone.

## Remaining project boundary

V5 remains the current engineering baseline for the original project claim:

```text
context RGB + calibrated cameras
-> target-independent typed metric 3D state
-> fixed zero-parameter renderer
-> RGB / depth / normal / point / visibility / evidence risk
```

A future sparse mutual-triangulation operator would be a separate research branch, not V6a.2. It
would need context-only mutual epipolar correspondences, multi-view reprojection consistency, and a
label-blind surface audit before any training. That direction is not authorized by this failure
seal and was not implemented here.
