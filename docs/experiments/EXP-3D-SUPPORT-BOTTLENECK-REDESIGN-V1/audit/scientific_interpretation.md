# Scientific interpretation: support redesign diagnostics

**Volume-only supplies positive evidence; the combined predeclared experiment remains inconclusive.**
These are GT-privileged frozen-carrier interventions, not deployable methods or mathematical upper bounds.

| Intervention | Full-context gain | 95% scene CI | Positive / tie / negative scenes |
|---|---:|---|---|
| R0 | -0.095226 | [-0.186745, -0.008134] | 5/1/11 |
| ORACLE_VOLUME | +0.115349 | [+0.014210, +0.217646] | 11/0/6 |
| ORACLE_SUPPORT | -0.245025 | [-0.361812, -0.129884] | 3/1/13 |
| ORACLE_VOLUME_SUPPORT | +0.089622 | [-0.026782, +0.216702] | 11/0/6 |

## Positive volume evidence and its limits

Oracle volume alone restores positive full-context gain with pooled CI above zero; new-dev stratum also has positive CI. This supports a volume contribution and must not be hidden by the combined stop.

The previously exposed stratum is less stable than new redesign dev; both remain development evidence.

| Intervention | Absolute full-context AbsRel | Absolute improvement vs R0 [CI] | Anchor worsening vs R0 [CI] |
|---|---:|---|---:|
| R0 | 0.679826 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| ORACLE_VOLUME | 0.645657 | +0.034169 [-0.070911, +0.135352] | +0.176405 [+0.041960, +0.337828] |
| ORACLE_SUPPORT | 0.829626 | -0.149800 [-0.236929, -0.068526] | +0.000000 [+0.000000, +0.000000] |
| ORACLE_VOLUME_SUPPORT | 0.671384 | +0.008442 [-0.076392, +0.096578] | +0.176405 [+0.041960, +0.337828] |

For volume-only, 83.8% of the change in full-context gain comes from a worse anchor after changing bounds. The remainder is an absolute full-context improvement. Therefore positive context gain is not the same as equally large absolute task recovery.

## Concentration, sensitivity and association

- R0: top-1/top-3 shares of positive gains 57.5%/87.7%; leave-one-scene-out mean range [-0.11816232432817494, -0.07112594808579151].
- ORACLE_VOLUME: top-1/top-3 shares of positive gains 22.1%/48.9%; leave-one-scene-out mean range [0.08588209820289912, 0.1379386513042917].
- ORACLE_SUPPORT: top-1/top-3 shares of positive gains 52.4%/100.0%; leave-one-scene-out mean range [-0.2688585534411205, -0.21296057760476722].
- ORACLE_VOLUME_SUPPORT: top-1/top-3 shares of positive gains 25.7%/64.8%; leave-one-scene-out mean range [0.055154516441632245, 0.11421226851456942].

Positive-share concentration uses the sum of positive scene gains, not the signed net denominator. Full per-scene and stratum statistics, both concentration definitions, and LOSO values are retained in JSON. Scatter correlations are descriptive and do not identify causality.

## Why the stop is inconclusive

Combined two-view candidate fraction is 93.18%, but supported GT-surface fraction is 66.22%, below 75%. Frustum support is not occlusion visibility. Finite 8^3 geometry and 128 candidates do not guarantee ideal surface support.

Predeclared combined intervention has CI crossing zero and supported-surface coverage below adequacy. Do not switch to the better volume-only gate after seeing scores. GT-free real-scene runs and matched retraining remain NOT_RUN_ORACLE_STOP.

NOT_IDENTIFIED: bounds also change voxel resolution, normalized feature/spatial distributions, opacity/path lengths, and anchor prior rendering. Allocation changes learned spatial distribution and varies by context. Frozen weights cannot separate these from learned capacity.

It is unsupported to conclude either “support is useless” or “the network is the sole cause.” Volume contributes, allocation alone worsens the frozen model, and coverage/training-distribution/representation remain coupled. Final holdout stays closed; no Dynamic TTT is run.
