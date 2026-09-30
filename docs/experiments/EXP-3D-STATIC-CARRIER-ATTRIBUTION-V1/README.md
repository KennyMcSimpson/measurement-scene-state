# EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1

**STATIC_STATE_STATUS=NOT_ESTABLISHED.** The frozen 1000A+600B B-final carrier does not
qualify for unseen-scene geometry. Seven checkpoint-unseen scenes were evaluated;
full context is worse than anchor in six scenes and ties in one. No Dynamic TTT,
controller training, write-rule tuning or new history experiment was run.

## Frozen experiment and data

Carrier SHA256 `be7b8b6d2ef366cad245732b9ff227802db596da65082ac0e160f24541579e42` exactly matches the previous experiment.
8³ volume, 128 candidates, centered bounds, RGB preprocessing, calibrated metric
cameras, normalized-ray depth and 64 renderer samples remain unchanged. Old A/B
roles and primary frames 8/9 are retained. All 90 old static rows reproduce exactly
for RGB MSE/SSIM, depth AbsRel/RMSE/δ1, opacity and coverage, with identical query-camera hashes.

The prelocked eight candidates came only from the existing research-train and
Hypersim official-train pools. Seven qualified. ai_005_003 failed the pre-existing
10 mm native-position p95 check (11.24169 mm); this is a threshold failure, not proof
of a camera bug. It was retained and never replaced after selection. Frame quality
and A/B feasibility used only metadata, RGB/depth validity and cameras; no model score.
Download: 81,591,203 bytes using bounded HTTP ranges, no full archive.

Checkpoint-unseen IDs: ai_004_001, ai_006_001, ai_007_002, ai_008_001, ai_009_001, ai_010_001, ai_011_001. Training IDs are ai_001_001, ai_002_001,
ai_003_001, verified against manifest, logs and checkpoint provenance. Official
volume/archive/asset identities are distinct; supplier reuse across distinct assets
cannot be independently excluded. ai_007_002 has an older `previously_observed` flag:
it is unseen to this checkpoint, not claimed historically unobserved by all researchers.
No protected/test or official validation split was opened. See `scene_split.json`,
`unseen_independence_audit.json`, and the preserved candidate/data/role locks.

## Unseen qualification

All rows use scene-equal means; primary full-context error is the predeclared average
of A and B. Positive gain means anchor error minus full-context error. Bootstrap
resamples seven paired scenes, 10,000 draws, seed 20260927, percentile 95% CI.

| Method | Depth AbsRel ↓ | RGB MSE ↓ | SSIM ↑ | RMSE ↓ | δ1 ↑ | Opacity | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.823763 | 0.128941 | 0.191480 | 5.865831 | 0.019504 | 0.716053 | 0.857143 |
| B | 0.770695 | 0.129283 | 0.215105 | 5.612089 | 0.041148 | 0.754203 | 0.857143 |
| anchor | 0.607558 | 0.164936 | 0.208778 | 4.943225 | 0.124427 | 0.529970 | 0.857143 |
| prior | 0.463780 | 0.129822 | 0.211967 | 3.323784 | 0.364713 | 1.000000 | 1.000000 |
| wrong_scene | 0.787288 | 0.186328 | 0.190355 | 5.792789 | 0.031007 | 0.748061 | 0.782282 |
| Residual_A | 1.017591 | 0.180660 | 0.163723 | 5.771171 | 0.067175 | 0.716053 | 0.857143 |
| Residual_B | 0.687987 | 0.168964 | 0.171592 | 4.570538 | 0.082753 | 0.754203 | 0.857143 |
| Residual_shuffled | 0.725766 | 0.153042 | 0.200582 | 4.927888 | 0.066417 | 0.748061 | 0.782282 |
| Residual_zero | 0.684057 | 0.200499 | 0.178956 | 4.519792 | 0.138321 | 0.000000 | 0.000000 |
| feature_shuffle | 0.744901 | 0.151440 | 0.145412 | 5.523553 | 0.055337 | 0.646898 | 0.806982 |
| channel_permutation | 0.823763 | 0.128941 | 0.191480 | 5.865831 | 0.019504 | 0.716053 | 0.857143 |
| state_mask | 1.000000 | 0.294323 | 0.019008 | 6.630876 | 0.000000 | 0.000000 | 0.000000 |


Full-context gain: **-0.189671**, CI **[-0.291047, -0.091291]**.
Wrong-scene damage relative to A: **-0.036475**, CI
**[-0.158533, 0.080005]**. Both qualification requirements fail.
Every leave-one-scene-out full-context gain remains negative; this failure is not
caused by one scene. All states were sealed before query/GT access and served two
queries unchanged. That structural property alone does not establish useful state.

| Scene | A | B | Anchor | Wrong scene | Full-context gain | Residual A | Residual B |
|---|---:|---:|---:|---:|---:|---:|---:|
| ai_004_001 | 0.872529 | 0.685990 | 0.504599 | 0.854410 | -0.274660 | 0.644667 | 0.673116 |
| ai_006_001 | 0.753534 | 0.682637 | 0.340917 | 0.410490 | -0.377168 | 0.671818 | 0.715205 |
| ai_007_002 | 0.985657 | 0.980036 | 0.616009 | 0.819507 | -0.366837 | 3.058768 | 0.824788 |
| ai_008_001 | 0.743810 | 0.791511 | 0.669692 | 0.947339 | -0.097969 | 0.362371 | 0.406382 |
| ai_009_001 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.974588 | 0.974588 |
| ai_010_001 | 0.681346 | 0.532930 | 0.491118 | 0.646135 | -0.116020 | 0.663616 | 0.502079 |
| ai_011_001 | 0.729463 | 0.721761 | 0.630570 | 0.833132 | -0.095042 | 0.747311 | 0.719754 |


Metrics include every GT-valid depth pixel, including uncovered rays. Coverage means
opacity >1e-6, not true visibility. A/B have higher mean opacity than anchor, yet worse
depth. Since rendered depth is the unnormalized sum(weights * ray distance), opacity
can affect apparent accuracy; no claim of opacity-independent geometry success is
made. `opacity_diagnostic.json` reports separately labeled diagnostic normalization
and common-high-opacity checks. They do not replace the frozen primary metric.
In that diagnostic, A/B/anchor AbsRel is .805089/.748956/.544680: anchor still wins.
The joint high-opacity subset is empty for some scenes; its full-cohort macro is null.
Common-visible and volume/support bucket metrics are in `bootstrap_results.json` and
raw records, including empty-mask counts. Missing scenes are not silently removed
from regional macros. Per-object metrics are unavailable because object segmentation
was not acquired; per-support-bucket metrics are supplied as requested.

## Why did ai_003_001 fail?

1. **Why is query RGB black?** All published source JPEGs 0–15 are already exactly
   zero RGB. PIL/OpenCV agree; resizing and model/evaluator tensors preserve them.
   ROOT_CAUSE=SOURCE_FRAME_IS_BLACK. Official metadata includes these images as train,
   with no exclusion and a non-BAD camera. Why the published source is black—render
   failure, transition, or another source issue—remains **UNKNOWN**. It is not evidence
   of a decoder/range/preprocessing bug. ai003 was never removed.
2. **Why no A two-view support?** Projection of all 128 candidates through cameras
   0/1/2 gives 13 single-view candidates and **zero** candidates with >=2 supporting
   views. B has 3 such candidates. The full coordinate/projection/validity records are
   in `support_audit_ai003.json`. A+stream has 4, but that does not repair static A.
3. **Why A equals anchor?** The current fusion/completion path needs two-view support.
   With none, both materialize the same prior branch and give identical predictions.
   This is expected behavior of the frozen carrier, not a newly discovered code bug.
   Primary GT surfaces are 100% inside its volume; A-supported surface neighborhoods
   are 0%. Inside the box does not imply observed support.

Unseen geometry also reveals coverage failure: ai009 primary GT surfaces are 0%
inside the volume on both queries; ai007 query 15 is only 0.06% inside, and ai008
queries are 30.2% and 6.67% inside. Geometry feasibility of the context does not ensure
query coverage. These scenes stay in the locked primary cohort; no post-score
filtering or volume expansion is applied. Thus support design is an evidenced
bottleneck; representation and training contributions are not isolated.

Camera/depth audit found no convention/unit/resize bug: old projection roundtrip max
1.17e-11 pixels, ray-depth roundtrip max 3.55e-15 m; native GT positions satisfy the
locked check. PREVIOUS_DEPTH_METRIC_INVALID=false. Depth is metric ray distance;
opacity normalization is a separate readout question. All old scenes received the
same audit. See `depth_definition.md`, `geometry_audit.json`, `volume_coverage.json`.

## Residual readout and state-use attribution

The existing 308-parameter 14→16→4 residual head uses exactly the same frozen states
as R-Fixed. Adam lr=.001, clip=1; only original three training scenes, query frames
12–14 for fitting and 15 for train-side validation, A/B contexts. The initial
3,000-step pilot was still improving, so a documented budget-only amendment extended
fresh same-seed training to 30,000 **before any residual primary/unseen evaluation**.
First 3,000 logs match exactly. The fixed validation rule selected step **7,500**.
No unseen/8–9 label was used for residual training or selection.

Selected train-side validation objective: .303350 versus .375932 initially; terminal
30,000-step validation worsened to .423491. The final 3,000-step training window still
improved 1.45%; global convergence is not established. This control is trained and
validated, but its limited capacity/training does not rule out every possible readout.
Head linear operations: 11,796,480 FLOPs per 128×160 query (multiply+add=2; excludes
renderer/activation). Measured fixed 5.21 ms, residual 10.81 ms including state safety
checks; head alone .0478 ms. See training curves, audit and timing definitions.

Average A/B: Fixed 0.797229, Residual 0.852789; Residual−Fixed
0.055560, CI [-0.178267, 0.387806].
B improves by 0.082708, while A worsens by
0.193828; selecting B as a new primary method after
seeing this would be invalid. Residual A replacement/zero-state do not reliably hurt
performance and zero-state even has lower macro AbsRel. Query-conditioned prior or
bypass remains plausible; this does not establish useful scene-specific information
or a general fixed-decoder bottleneck. B-specific bypass was not separately tested.

Context-feature spatial shuffle, feature-channel reversal and zero-evidence state
masking are diagnostic only. Prediction RMS/hash changes, not just latent changes,
are saved. Channel reversal leaves fixed typed-field rendering unchanged by design;
its residual effect must not be read as a fixed-renderer bug. No state was mutated.

## Train-query versus heldout-query gap

A AbsRel: 0.331528 on previously used 12–15 versus 2.213562 on 8–9. The macro gap
1.882034 is 94.59% contributed by ai003. Both ai003 cohorts are black and have zero A
support, so blackness alone cannot explain their difference. Heldout versus training
nearest-A view angle is 60.38° versus 20.74°, and approximate common visibility is
.3305 versus .8700. ai002 heldout volume coverage is .5228 versus .8846, with supported
surface neighborhood .1097 versus .2170. This supports view/support and coverage
limitations; it does not isolate training memorization. ai001 has a remaining
unexplained gap. Full per-frame baseline, view, depth, validity and overlap statistics
are in `train_heldout_gap_diagnosis.json` and its narrative report.

## Reproduction, tests and limitations

Run `commands.sh summary` from the repository root for raw-only statistics, or
`commands.sh full` with a fresh output directory for downloads, training, inference,
scripted audits and tests. The full mode ends with regenerated statistics; it does
not recreate the manually collected official-source-validity notes or the curated
environment/test/provenance report. Those original records are archived separately;
new repetitions must record their own environment and execution results. Archived
locks/raw reproduce all summaries via `scripts/report_static_attribution.py`. Tests:
**610 passed, 1 skipped**. The single historical V5 checkpoint test is skipped because that
optional checkpoint is unavailable. New tests cover RGB preservation, camera/depth
roundtrips, scene/query isolation, residual immutability, unchanged replacement camera,
scene bootstrap and score-independent quality/volume auditing. Full log: `tests.log`.

For a clean checkout containing only the published report, recompute statistics with:

```bash
.venv/bin/python scripts/report_static_attribution.py --raw-only   --output docs/experiments/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1   --docs /tmp/static-attribution-recomputed
```

Raw model results are in `static_results.json`/CSV; all official statistics can be
regenerated without rerunning inference. `artifacts/` stores deterministic gzip copies
of detailed audit/lock/training records; `artifact_manifest.json` maps them to output
paths with original SHA256. Local full states, media, native geometry arrays and both
residual checkpoints remain under outputs. Checkpoints/data are not duplicated into
Git docs; downloading requires the published Hypersim source and original carrier
artifacts described in the previous training report. `commands.sh` downloads pinned
metadata and invokes bounded data preparation. Source asset identity is audited but
not a forensic guarantee against supplier reuse. Seven dev scenes and sparse candidate
coverage limit scope. Negative qualification is not a proof that all shared states or
all visible signals are impossible. **Dynamic TTT remains closed for this carrier.**
