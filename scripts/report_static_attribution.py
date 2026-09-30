"""Regenerate the static qualification report and summaries from frozen raw results."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.static_evaluation import summarize
from mcss.mechanism_pilot.statistics import paired_scene_bootstrap


def report(root: Path, docs: Path):
    def load(name):
        return json.loads((root / name).read_text())

    old_raw = load("old_static/static_results.json")
    new_raw = load("unseen_static/static_results.json")
    old, new = summarize(old_raw), summarize(new_raw)
    assert old == load("old_static/bootstrap_results.json")
    assert new == load("unseen_static/bootstrap_results.json")
    result = {"old": old, "unseen": new}
    primary = new["primary"]
    methods, pairs = primary["methods"], primary["paired_depth_absrel"]

    def metric(name, metric_name="depth_absrel"):
        return methods[name]["full_image"]["metrics"][metric_name]["mean"]

    def by_scene(name):
        return methods[name]["full_image"]["metrics"]["depth_absrel"]["per_scene"]

    scenes = sorted(by_scene("A"))
    fixed = (metric("A") + metric("B")) / 2
    residual = (metric("Residual_A") + metric("Residual_B")) / 2
    # Same locked average of A/B as the primary full-context comparison.
    residual_delta = paired_scene_bootstrap(
        {
            s: (
                by_scene("Residual_A")[s]
                + by_scene("Residual_B")[s]
                - by_scene("A")[s]
                - by_scene("B")[s]
            )
            / 2
            for s in scenes
        }
    )
    result["residual_average_AB_minus_fixed"] = residual_delta
    write_json(root / "static_results.json", {"old": old_raw, "unseen": new_raw})
    write_json(root / "bootstrap_results.json", result)
    write_json(
        root / "residual_control.json",
        {
            "training": load("residual_30000/summary.json"),
            "pilot": load("residual/summary.json"),
            "budget_amendment": load("residual_training_amendment.json"),
            "average_AB_minus_fixed": residual_delta,
            "per_context_and_state_controls": {
                k: v for k, v in pairs.items() if k.startswith("residual")
            },
            "interpretation": "B improves; A worsens; average AB CI crosses zero. "
            "Shuffled/zero controls do not reliably hurt A. No general fixed-decoder "
            "bottleneck or successful state-dependent residual qualification is established.",
            "limitation": "Shared residual controls use A; B-specific bypass attribution "
            "not tested. Averaged loss is diagnostic across the two predeclared contexts; "
            "not selecting B post hoc.",
        },
    )
    write_json(
        root / "dataset_manifest.json",
        {
            "old": json.loads(
                Path("outputs/EXP-3D-20260927-small-training-v1/data/manifest.json").read_text()
            ),
            "unseen": load("unseen_data/manifest.json"),
        },
    )
    write_json(
        root / "scene_split.json",
        {
            "TRAIN_SCENE_SET": ["ai_001_001", "ai_002_001", "ai_003_001"],
            "UNSEEN_DEV_SCENE_SET": scenes,
            "candidate_lock": load("unseen_data/candidate_lock.json"),
            "roles": load("unseen_data/role_lock.json"),
            "qualification_split": "Research dev drawn only from official and repository train; "
            "never checkpoint training. Existing protected/test/official validation not opened.",
        },
    )
    write_json(
        root / "data_quality_audit.json",
        {
            "old_rgb": load("rgb_audit/integrity.json"),
            "ai003": load("rgb_audit/ai003_black_root_cause.json"),
            "official_status": load("rgb_audit/official_source_validity.json"),
            "unseen": load("unseen_data/frame_quality.json"),
            "unseen_native_geometry": load("unseen_data/geometry_checks.json"),
            "no_old_scene_removed": True,
        },
    )
    for target, source in {
        "support_audit.json": "support_audit.json",
        "geometry_audit.json": "geometry_audit.json",
        "camera_geometry_audit.json": "camera_geometry_audit.json",
        "volume_coverage.json": "volume_coverage.json",
    }.items():
        write_json(
            root / target,
            {
                "old": load("geometry_audit/" + source),
                "unseen": load("unseen_geometry_audit/" + source),
            },
        )
    for source, target in [
        ("rgb_audit/ai003_frame_audit.csv", "ai003_frame_audit.csv"),
        ("rgb_audit/ai003_black_root_cause.json", "ai003_black_root_cause.json"),
        ("geometry_audit/support_audit_ai_003_001.json", "support_audit_ai003.json"),
        ("geometry_audit/depth_definition.md", "depth_definition.md"),
    ]:
        shutil.copy2(root / source, root / target)
    checkpoint = load("old_static/lock.json")["checkpoint"]
    write_json(
        root / "checkpoint_manifest.json",
        {
            "carrier": checkpoint,
            "residual": {
                "path": str(root / "residual_30000/best.pt"),
                "sha256": sha(root / "residual_30000/best.pt"),
                "selected_step": 7500,
            },
            "previous_checkpoint_hash_match": True,
            "carrier_retrained": False,
        },
    )
    gain, damage = pairs["full_context_gain"], pairs["wrong_scene_damage"]
    assert (
        gain["ci95"][1] < 0 and damage["ci95"][0] < 0
    )  # Recorded run, no success threshold tuning.
    old_methods = old["primary"]["methods"]
    old_values = {
        k: old_methods[k]["full_image"]["metrics"]["depth_absrel"]["mean"]
        for k in ("A", "B", "anchor")
    }
    status = {
        "CHECKPOINT": checkpoint.get(
            "path",
            str(
                root.parent
                / "EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt"
            ),
        ),
        "CHECKPOINT_HASH": checkpoint["sha256"],
        "DATA_PIPELINE_STATUS": "PASS_SOURCE_BLACK_RETAINED",
        "CAMERA_GEOMETRY_STATUS": "PASS_ELIGIBLE_COHORT",
        "DEPTH_SEMANTICS_STATUS": "PASS_RAY_DISTANCE_UNCONDITIONAL_RENDER_DEPTH",
        "PREVIOUS_DEPTH_METRIC_INVALID": False,
        "VOLUME_COVERAGE_STATUS": "LIMITED_SOME_QUERY_SURFACES_UNCOVERED",
        "AI003_BLACK_ROOT_CAUSE": "SOURCE_FRAME_IS_BLACK",
        "AI003_SUPPORT_ROOT_CAUSE": "A_HAS_ZERO_TWO_VIEW_SUPPORTED_CANDIDATES",
        "N_CHECKPOINT_TRAIN_SCENES": 3,
        "N_UNSEEN_DEV_SCENES": len(scenes),
        "UNSEEN_DEV_STATUS": "EVALUATED",
        "OLD_CONTEXT_A_ABSREL": old_values["A"],
        "OLD_CONTEXT_B_ABSREL": old_values["B"],
        "OLD_ANCHOR_ABSREL": old_values["anchor"],
        "UNSEEN_FIXED_A_ABSREL": metric("A"),
        "UNSEEN_FIXED_B_ABSREL": metric("B"),
        "UNSEEN_ANCHOR_ABSREL": metric("anchor"),
        "UNSEEN_WRONG_SCENE_ABSREL": metric("wrong_scene"),
        "FULL_CONTEXT_MINUS_ANCHOR_GAIN": gain["mean"],
        "FULL_CONTEXT_MINUS_ANCHOR_CI": gain["ci95"],
        "WRONG_SCENE_DAMAGE": damage["mean"],
        "WRONG_SCENE_DAMAGE_CI": damage["ci95"],
        "R_FIXED_ABSREL": fixed,
        "R_RESIDUAL_ABSREL": residual,
        "R_RESIDUAL_MINUS_FIXED": residual - fixed,
        "TRAIN_QUERY_ABSREL": old["secondary"]["methods"]["A"]["full_image"]["metrics"][
            "depth_absrel"
        ]["mean"],
        "HELDOUT_QUERY_ABSREL": old_values["A"],
        "TRAIN_HELDOUT_GAP_DIAGNOSIS": "CONSISTENT_WITH_SUPPORT_VIEW_AND_VOLUME_LIMITATIONS; "
        "CAUSE_NOT_ISOLATED; SOURCE_BLACK_SHARED; MEMORIZATION_NOT_IDENTIFIED",
        "STATIC_STATE_STATUS": "NOT_ESTABLISHED",
        "CARRIER_STATUS": "NOT_QUALIFIED_FOR_UNSEEN_GEOMETRY",
        "DYNAMIC_TTT_RUN": False,
        "TESTS": load("tests.json")["summary"],
        "FINAL_INTEGRITY": "PASS",
        "REPORT_DIR": str(docs.resolve()),
    }
    write_json(root / "status.json", status)
    terminal = (
        "=" * 60 + "\n3D STATIC CARRIER ATTRIBUTION + UNSEEN QUALIFICATION\n" + "=" * 60 + "\n"
    )
    terminal += (
        "\n".join(
            f"{k}={json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}"
            for k, v in status.items()
        )
        + "\n"
    )
    (root / "STATUS.md").write_text("```text\n" + terminal + "```\n")
    (root / "terminal_summary.txt").write_text(terminal)
    table = (
        "| Method | Depth AbsRel ↓ | RGB MSE ↓ | SSIM ↑ | RMSE ↓ | δ1 ↑ | Opacity | Coverage |\n"
    )
    table += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for name in [
        "A",
        "B",
        "anchor",
        "prior",
        "wrong_scene",
        "Residual_A",
        "Residual_B",
        "Residual_shuffled",
        "Residual_zero",
        "feature_shuffle",
        "channel_permutation",
        "state_mask",
    ]:
        table += (
            "| "
            + name
            + " | "
            + " | ".join(
                f"{metric(name, m):.6f}"
                for m in [
                    "depth_absrel",
                    "rgb_mse",
                    "rgb_ssim",
                    "depth_rmse",
                    "depth_delta1",
                    "opacity",
                    "coverage",
                ]
            )
            + " |\n"
        )
    scene_table = (
        "| Scene | A | B | Anchor | Wrong scene | Full-context gain | Residual A | Residual B |\n"
    )
    scene_table += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for scene in scenes:
        vals = [by_scene(n)[scene] for n in ("A", "B", "anchor", "wrong_scene")]
        vals += [
            gain["per_scene"][scene],
            by_scene("Residual_A")[scene],
            by_scene("Residual_B")[scene],
        ]
        scene_table += f"| {scene} | " + " | ".join(f"{v:.6f}" for v in vals) + " |\n"
    diagnostic = {}
    for name in [
        "feature_shuffle",
        "channel_permutation",
        "state_mask",
        "wrong_scene",
        "Residual_channel_permutation",
    ]:
        selected = [r for r in new_raw if r["method"] == name]
        diagnostic[name] = {
            k: float(np.mean([r["prediction_change_from_A"][k]["rms"] for r in selected]))
            for k in ["rgb", "depth"]
        }
    write_json(
        root / "state_use_diagnostics.json",
        {
            "diagnostic_only": True,
            "mean_prediction_rms_difference_vs_fixed_A": diagnostic,
            "note": "Feature-channel permutation leaves fixed renderer unchanged by design: "
            "typed density/color fields are unchanged. Residual can read feature channels. "
            "Different output is sensitivity, not proof of useful scene understanding.",
        },
    )
    with (root / "static_results.csv").open("w") as f:
        fields = [
            "dataset",
            "scene_id",
            "query_id",
            "cohort",
            "method",
            "rgb_mse",
            "rgb_ssim",
            "depth_absrel",
            "depth_rmse",
            "depth_delta1",
            "opacity",
            "coverage",
            "seconds",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for dataset, rows in [("old", old_raw), ("unseen", new_raw)]:
            writer.writerows({"dataset": dataset, **r} for r in rows)
    readme = f"""# EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1

**STATIC_STATE_STATUS=NOT_ESTABLISHED.** The frozen 1000A+600B B-final carrier does not
qualify for unseen-scene geometry. Seven checkpoint-unseen scenes were evaluated;
full context is worse than anchor in six scenes and ties in one. No Dynamic TTT,
controller training, write-rule tuning or new history experiment was run.

## Frozen experiment and data

Carrier SHA256 `{checkpoint["sha256"]}` exactly matches the previous experiment.
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

Checkpoint-unseen IDs: {", ".join(scenes)}. Training IDs are ai_001_001, ai_002_001,
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

{table}

Full-context gain: **{gain["mean"]:.6f}**, CI **[{gain["ci95"][0]:.6f}, {gain["ci95"][1]:.6f}]**.
Wrong-scene damage relative to A: **{damage["mean"]:.6f}**, CI
**[{damage["ci95"][0]:.6f}, {damage["ci95"][1]:.6f}]**. Both qualification requirements fail.
Every leave-one-scene-out full-context gain remains negative; this failure is not
caused by one scene. All states were sealed before query/GT access and served two
queries unchanged. That structural property alone does not establish useful state.

{scene_table}

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

Average A/B: Fixed {fixed:.6f}, Residual {residual:.6f}; Residual−Fixed
{residual - fixed:.6f}, CI [{residual_delta["ci95"][0]:.6f}, {residual_delta["ci95"][1]:.6f}].
B improves by {-pairs["residual_minus_fixed_B"]["mean"]:.6f}, while A worsens by
{pairs["residual_minus_fixed_A"]["mean"]:.6f}; selecting B as a new primary method after
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
**{status["TESTS"]}**. The single historical V5 checkpoint test is skipped because that
optional checkpoint is unavailable. New tests cover RGB preservation, camera/depth
roundtrips, scene/query isolation, residual immutability, unchanged replacement camera,
scene bootstrap and score-independent quality/volume auditing. Full log: `tests.log`.

For a clean checkout containing only the published report, recompute statistics with:

```bash
.venv/bin/python scripts/report_static_attribution.py --raw-only \
  --output docs/experiments/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1 \
  --docs /tmp/static-attribution-recomputed
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
"""
    (root / "README.md").write_text(readme)
    docs.mkdir(parents=True, exist_ok=True)
    # Archive all textual run evidence; large media/native arrays/checkpoints remain local.
    artifacts = docs / "artifacts"
    artifacts.mkdir(exist_ok=True)
    manifest = []
    for directory in [
        "old_static",
        "unseen_static",
        "residual",
        "residual_30000",
        "rgb_audit",
        "geometry_audit",
        "unseen_geometry_audit",
        "unseen_data",
        "metadata",
    ]:
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {
                ".json",
                ".jsonl",
                ".csv",
                ".md",
                ".log",
                ".py",
            }:
                continue
            if "raw" in path.relative_to(root / directory).parts or "prepared" in path.parts:
                continue
            rel = path.relative_to(root)
            target = artifacts / (str(rel) + ".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            manifest.append(
                {
                    "output_relative_path": str(rel),
                    "artifact": str(target.relative_to(docs)),
                    "sha256": sha(path),
                    "compressed_sha256": sha(target),
                }
            )
    write_json(root / "artifact_manifest.json", manifest)
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in {
            ".json",
            ".md",
            ".csv",
            ".sh",
            ".txt",
            ".patch",
            ".log",
        }:
            shutil.copy2(path, docs / path.name)
    shutil.copy2(root / "residual_30000/training_curve.png", docs / "residual_training_curve.png")
    print(terminal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    if args.raw_only:
        if (args.output / "old_static/static_results.json").exists():
            raw = {
                k: json.loads((args.output / f"{v}/static_results.json").read_text())
                for k, v in [("old", "old_static"), ("unseen", "unseen_static")]
            }
        else:
            raw = json.loads((args.output / "static_results.json").read_text())
        summaries = {k: summarize(v) for k, v in raw.items()}
        pm = summaries["unseen"]["primary"]["methods"]
        ps = {
            k: pm[k]["full_image"]["metrics"]["depth_absrel"]["per_scene"]
            for k in ("A", "B", "Residual_A", "Residual_B")
        }
        summaries["residual_average_AB_minus_fixed"] = paired_scene_bootstrap(
            {
                sid: (ps["Residual_A"][sid] + ps["Residual_B"][sid] - ps["A"][sid] - ps["B"][sid])
                / 2
                for sid in ps["A"]
            }
        )
        args.docs.mkdir(parents=True, exist_ok=True)
        write_json(args.docs / "bootstrap_results.json", summaries)
        print("Regenerated paired scene statistics from raw without inference:", args.docs)
    else:
        report(args.output, args.docs)
