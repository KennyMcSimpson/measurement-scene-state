# V6a.1 Fixed Ray-Evidence Transport Design

## Decision and scope

V6a step 5000 failed its frozen one-window capacity gate only on depth: RGB PSNR changed by
`-0.0473 dB`, normal mean angle improved by `4.82%`, and depth AbsRel worsened by `7.878%`
against the V5 step-4500 reference, beyond the admitted `5%`. V6b and the 16/8 pilot remain
blocked.

The context-only telemetry report at
`outputs/hypersim_er_v6a_dual_evidence_telemetry_step5000_v2/evidence_telemetry.json` shows that
the center candidate is not merely weak. With score normalization by actual valid observers and
at least two other observers, top-appearance profiles retain `93.07%` of positive excess on
off-center candidates. The center fractional argmax rate is `7.59%` versus a `21.09%` chance
baseline. This authorizes one new mechanism branch that transports fixed evidence to the 3D
candidate that generated it. It does not validate the transported evidence in advance.

V5 `evidence_residual` and V6a `dual_evidence` remain immutable ablations. V6a.1 uses the explicit
architecture name `dual_evidence_transport`. It preserves:

```text
context RGB + calibrated context cameras
-> one target-independent typed metric 3D state
-> fixed zero-parameter renderer
-> RGB / depth / normal / point / visibility / evidence risk
```

It adds no target encoder, target-conditioned state, learned renderer, image warp, Gaussian
primitive, per-scene optimization, or target-label input.

## Fixed local competition

For source voxel center `x_n`, reference view `i`, offsets `r in {-2,-1,0,1,2}`, and the existing
minimum native voxel edge `delta`, candidates are:

```text
y_inr = x_n + r * delta * d_i(x_n)
```

The fixed RGB-pyramid pair score is unchanged, but each candidate score is divided by its actual
valid observer count. A candidate requires the reference projection and at least two other valid
observers. A profile contributes only when at least three of five candidates are valid. Two-view
inputs therefore produce zero transport localization support.

For valid candidates:

```text
p_inr = softmax_valid(score_inr / 0.05)
m_inr = clamp((K * p_inr - 1) / (K - 1), 0, 1)
```

Flat profiles have zero transported mass. Temperature, offsets, descriptor, and observer minimum
are fixed before any target-label audit.

## Trilinear transport

Each candidate `y_inr` is converted to native voxel-center coordinates:

```text
g(y) = (y - b_min) / ((b_max - b_min) / [W,H,D]) - 0.5
```

Its mass is splatted to the eight neighboring cells. Neighbors outside the AABB are discarded and
the remaining weights are renormalized per candidate. Let `a_n` be detached V5 appearance
confidence at the source voxel and `w_inrq` the normalized splat weight:

```text
R(q) = sum w_inrq * valid_inr
A(q) = sum w_inrq * valid_inr * a_n
M(q) = sum w_inrq * valid_inr * a_n * m_inr

c_app_transport(q) = A(q) / R(q) when R(q) > 0 else 0
peak_transport(q)  = M(q) / A(q) when A(q) > 0 else 0
c_loc_transport(q) = M(q) / R(q) when R(q) > 0 else 0
```

The implementation must satisfy
`c_loc_transport == c_app_transport * peak_transport` within FP32 tolerance and clamp all three
confidence fields to `[0,1]`. `R` is exposed only as nonnegative transport coverage telemetry.

## State and reconstruction contract

V6a.1 adds `TransportEvidence`; it does not change `DualEvidence` semantics. Geometry uses:

```text
base_density_logits = 4 * c_loc_transport - 2
localization_gate = floor + (1 - floor) * (1 - c_loc_transport)
density_logits = base_density_logits + localization_gate * density_residual
```

The V5 appearance path remains unchanged:

```text
appearance_confidence = native/high-resolution V5 appearance agreement
appearance_gate = floor + (1 - floor) * (1 - appearance_confidence)
color = sigmoid(logit(base_color) + appearance_gate * color_logit_residual)
```

The trainable parameter names and count must remain identical to V5 and V6a. A V6a.1 run starts
from seed 17 rather than resuming a V5/V6a checkpoint.

## Required evidence and stopping rules

Static acceptance requires mass-conserving boundary splat, off-center displacement, flat-profile
zero support, reconstruction identities, detached evidence, view-permutation invariance, unchanged
V5/V6a behavior, zero renderer parameters, CPU and BF16 CUDA finite forward/backward, full pytest,
Ruff, and compileall.

V6a.1 then runs the same 5,000-step one-window gate against V5 step 4500:

- PSNR drop no worse than `0.20 dB`;
- depth AbsRel relative worsening no greater than `5%`;
- normal mean angle relative worsening no greater than `5%`.

If this capacity gate fails, the transport branch stops. If it passes, run the frozen 32-window
validation surface audit before any V6b training. V6a.1 must improve scene-macro surface-vs-free
and surface-vs-behind AUROC by at least `0.05` over V5, lower support in both non-surface bins,
retain nontrivial support coverage, and make risk increase with held-out geometric error. Failure
blocks V6b and long training. No external-baseline or CVPR-ready claim is allowed before matched
multi-scene and external evaluations.

