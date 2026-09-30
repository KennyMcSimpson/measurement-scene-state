# Train-query versus frame-heldout gap

Static-only attribution using existing audited data and saved predictions; no model rerun.

Context A scene-macro AbsRel: training queries 0.331528, heldout queries 2.213562; gap 1.882034. Both use the same three TRAIN scenes.

## ai_001_001

| Quantity | Heldout 8,9 | Training 12–15 |
|---|---:|---:|
| saved_context_A_depth_absrel | 0.549785 | 0.416402 |
| anchor_camera_baseline_m | 2.742613 | 2.544937 |
| anchor_view_angle_degrees | 73.605003 | 86.260467 |
| context_a_minimum_camera_baseline_m | 2.577006 | 2.201028 |
| context_a_minimum_view_angle_degrees | 29.926526 | 31.035272 |
| depth_mean_m | 3.651985 | 2.685198 |
| mean_frame_depth_median_m | 3.906421 | 2.869238 |
| mean_frame_depth_p95_m | 4.440337 | 3.927791 |
| inside_volume_fraction | 1.000000 | 1.000000 |
| valid_GT_fraction | 1.000000 | 1.000000 |
| A_supported_surface_fraction | 0.000000 | 0.000000 |
| B_supported_surface_fraction | 0.021753 | 0.012134 |
| approx_depth_consistent_common_fraction | 0.496631 | 0.397839 |
| source_RGB_zero_pixel_fraction | 0.000000 | 0.000000 |

A/B >=2-view supported candidates: 4/8 of 128. Gap contribution: 2.36%.

Observed contributors: SUPPORT FAILURE. Gap attribution: UNKNOWN.

Only 4/128 A candidates have >=2-view support. Neither cohort has a GT surface point within the fixed neighborhood of an A-supported candidate. Both cohorts lie inside the volume; heldout views are not farther in orientation from anchor. Sparse support is shared by both cohorts, so it does not uniquely explain the gap. Depth/view distributions differ; memorization remains untested.

## ai_002_001

| Quantity | Heldout 8,9 | Training 12–15 |
|---|---:|---:|
| saved_context_A_depth_absrel | 0.308333 | 0.136033 |
| anchor_camera_baseline_m | 1.233221 | 2.935173 |
| anchor_view_angle_degrees | 13.490299 | 69.528939 |
| context_a_minimum_camera_baseline_m | 0.905922 | 2.273306 |
| context_a_minimum_view_angle_degrees | 6.666488 | 69.528939 |
| depth_mean_m | 5.349388 | 3.981247 |
| mean_frame_depth_median_m | 5.508057 | 3.936926 |
| mean_frame_depth_p95_m | 6.699189 | 5.459033 |
| inside_volume_fraction | 0.522803 | 0.884570 |
| valid_GT_fraction | 1.000000 | 1.000000 |
| A_supported_surface_fraction | 0.109668 | 0.216980 |
| B_supported_surface_fraction | 0.088599 | 0.000000 |
| approx_depth_consistent_common_fraction | 0.847778 | 0.350549 |
| source_RGB_zero_pixel_fraction | 0.000441 | 0.000002 |

A/B >=2-view supported candidates: 6/7 of 128. Gap contribution: 3.05%.

Observed contributors: VOLUME COVERAGE, SUPPORT FAILURE. Gap attribution: VOLUME COVERAGE.

Heldout surfaces lie outside the modeled volume more often and have less A-supported neighborhood coverage, despite heldout cameras being closer to anchor and less rotated. This supports a coverage-distribution contribution, not a proven sole cause or simple farther-view extrapolation.

## ai_003_001

| Quantity | Heldout 8,9 | Training 12–15 |
|---|---:|---:|
| saved_context_A_depth_absrel | 5.782568 | 0.442149 |
| anchor_camera_baseline_m | 0.319515 | 0.450772 |
| anchor_view_angle_degrees | 60.375138 | 107.600998 |
| context_a_minimum_camera_baseline_m | 0.268597 | 0.250297 |
| context_a_minimum_view_angle_degrees | 60.375137 | 20.738141 |
| depth_mean_m | 0.679507 | 0.689735 |
| mean_frame_depth_median_m | 0.572693 | 0.505569 |
| mean_frame_depth_p95_m | 1.342413 | 1.616979 |
| inside_volume_fraction | 1.000000 | 1.000000 |
| valid_GT_fraction | 1.000000 | 1.000000 |
| A_supported_surface_fraction | 0.000000 | 0.000000 |
| B_supported_surface_fraction | 0.000000 | 0.120386 |
| approx_depth_consistent_common_fraction | 0.330469 | 0.869983 |
| source_RGB_zero_pixel_fraction | 1.000000 | 1.000000 |

A/B >=2-view supported candidates: 0/3 of 128. Gap contribution: 94.59%.

Observed contributors: DATA QUALITY, SUPPORT FAILURE. Gap attribution: UNKNOWN.

All source RGB frames are black and A has zero >=2-view candidate support in both cohorts. A equals anchor under this architecture. Both cohorts' GT surfaces lie inside the volume; invalid GT and outside-volume geometry do not explain the gap. Query orientations, context overlap and per-frame depth distributions differ. No controlled experiment separates learned prior/view dependence from training memorization; black source alone does not identify why only heldout depth collapses.

## Limits

- No causal overfitting conclusion from train/heldout score gap alone.
- Both query cohorts belong to carrier training scenes; frame-heldout is not unseen-scene validation.
- Candidate-neighborhood support is a fixed geometric proxy, not a test of all learned receptive fields.
- Context overlap uses approximate nearest-depth consistency, not exact visibility.
- Depth medians/p95 shown in means are means of frame quantiles, not pooled-pixel quantiles.
- Saved static scores are previous locked B-final outputs; no rerender or checkpoint selection here.
