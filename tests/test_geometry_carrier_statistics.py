"""Synthetic matched carrier DEV gates, never media or model predictions."""

import copy
import json

import pytest

from mcss.mechanism_pilot.geometry_carrier_statistics import METHODS, METRICS, analyze


def fixture():
    ids, seeds = [f"s{i}" for i in range(8)], [20260928, 20260929]
    manifest = {
        "scenes": [
            {
                "scene_id": s,
                "split": "DEV",
                "roles": {"primary_query": [8, 9], "context_a": [0, 1, 2], "context_b": [0, 3, 4]},
            }
            for s in ids
        ],
        "fresh_status": "BLOCKED_INDEPENDENCE_UNRESOLVED",
    }
    query, context, matched, static = [], [], [], []
    for scene in ids:
        for seed in seeds:
            for role in ("A", "B"):
                for variant in ("C0", "C1"):
                    value = (
                        0.4 if variant == "C0" else 0.39
                    )  # Below old .02 threshold deliberately.
                    for fid in (8, 9):
                        base = dict(
                            scene_id=scene, seed=seed, role=role, variant=variant, query_id=fid
                        )
                        for method in METHODS:
                            error = value + (0 if method == "direct" else 0.1)
                            query.append(
                                {
                                    **base,
                                    "method": method,
                                    **{m: error for m in METRICS},
                                    "coverage": 1.0,
                                    "opacity": 0.8,
                                    "ray_hitfraction": 1.0,
                                }
                            )
                        matched.append(
                            {
                                **base,
                                "method": "direct",
                                "depth_absrel": value,
                                "normalized_depth_absrel": value,
                                "common_valid_count": 9,
                                "total_gt_valid": 9,
                                "common_mask_hash": "a" * 64,
                            }
                        )
                        static.append(
                            {
                                **base,
                                "comparison": "anchor_minus_direct",
                                "anchor_depth_absrel": value + 0.1,
                                "direct_depth_absrel": value,
                                "anchor_normalized_depth_absrel": value + 0.1,
                                "direct_normalized_depth_absrel": value,
                                "common_valid_count": 9,
                                "total_gt_valid": 9,
                                "common_mask_hash": "c" * 64,
                            }
                        )
                    for fid in [0, 1, 2] if role == "A" else [0, 3, 4]:
                        context.append(
                            dict(
                                scene_id=scene,
                                seed=seed,
                                role=role,
                                variant=variant,
                                method="direct",
                                frame_id=fid,
                                **{m: 0.1 for m in METRICS},
                            )
                        )
    audit = dict(
        status="PASS",
        shared_state_multiple_queries=True,
        query_after_state_seal=True,
        recipient_camera_preserved=True,
    )
    return (
        query,
        context,
        matched,
        manifest,
        dict(seeds=seeds, state_use_audit=audit, static_matched_rows=static, draws=100),
    )


def run(data):
    q, c, m, manifest, kwargs = data
    return analyze(q, c, m, manifest, **kwargs)


def test_positive_gain_under_old_threshold_and_fresh_remains_closed():
    result = run(fixture())
    q = result["qualification_results"]
    assert q["SURFACE_TRAINING_STATUS"] == "SUPPORTED"
    assert q["STATIC_DEV_STATUS"] == "SUPPORTED"
    assert result["static_results"]["surface_gain"]["mean"] == pytest.approx(0.01)
    assert not q["FRESH_QUALIFICATION_OPENED"] and not q["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    assert q["FINAL_STATIC_STATUS"] == "NOT_ESTABLISHED"
    assert q["SEED_ROBUSTNESS"] == "DIRECTION_REPLICATED"


def test_no_hit_scene_kept_in_overall_and_common_mask_null():
    data = fixture()
    for row in data[0]:
        if row["scene_id"] == "s0":
            row.update(coverage=0.0, ray_hitfraction=0.0)
    for row in data[2]:
        if row["scene_id"] == "s0":
            row.update(common_valid_count=0, depth_absrel=None, normalized_depth_absrel=None)
    result = run(data)
    state = result["state_use_results"]
    assert state["opacity_artifact_gate"]
    assert state["no_hit_scenes"] == ["s0"]
    assert state["common_mask_gains"]["depth_absrel"]["scene_coverage"] == 7 / 8
    assert state["common_mask_gains"]["depth_absrel"]["per_scene"]["s0"] is None
    assert result["static_results"]["surface_gain"]["n_scenes"] == 8


def test_insufficient_common_coverage_or_normalized_gain_blocks_support():
    data = fixture()
    for row in data[2]:
        if row["variant"] == "C1":
            row["normalized_depth_absrel"] = 0.4
    result = run(data)
    assert not result["state_use_results"]["opacity_artifact_gate"]
    assert result["qualification_results"]["SURFACE_TRAINING_STATUS"] == "PARTIAL"
    data = fixture()
    for row in data[2]:
        if row["scene_id"] in ("s0", "s1", "s2"):
            row.update(common_valid_count=0, depth_absrel=None, normalized_depth_absrel=None)
    assert not run(data)["state_use_results"]["opacity_artifact_gate"]


def test_seed_average_not_pseudo_replication_and_negative_seed_reported():
    data = fixture()
    for row in data[0]:
        if row["variant"] == "C1" and row["seed"] == 20260929:
            row["depth_absrel"] += 0.03
    result = run(data)
    assert result["static_results"]["surface_gain"]["n_scenes"] == 8
    assert result["static_results"]["surface_gain"]["mean"] == pytest.approx(-0.005)
    assert result["qualification_results"]["SURFACE_TRAINING_STATUS"] == "HARMFUL"
    assert result["qualification_results"]["SEED_ROBUSTNESS"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "train", "mask", "hit"])
def test_roster_and_common_mask_fail_closed(fault):
    data = fixture()
    if fault == "missing":
        data[0].pop()
    if fault == "duplicate":
        data[0].append(copy.deepcopy(data[0][0]))
    if fault == "train":
        data[3]["scenes"][0]["split"] = "TRAIN"
    if fault == "mask":
        data[2][0]["common_mask_hash"] = "b" * 64
    if fault == "hit":
        for row in data[0]:
            if row["variant"] == "C1":
                row["ray_hitfraction"] = 0.8
    with pytest.raises((ValueError, PermissionError)):
        run(data)


def test_state_integrity_required_and_raw_roundtrip_deterministic():
    data = fixture()
    first = run(data)
    restored = json.loads(json.dumps(data))
    assert run(restored) == first
    data[4]["state_use_audit"]["shared_state_multiple_queries"] = False
    assert run(data)["qualification_results"]["STATIC_DEV_STATUS"] == "PARTIAL"


def region_fixture(data):
    output = []
    for row in data[0]:
        if row["method"] != "direct":
            continue
        for region, count in [
            ("OBS0", 0 if row["scene_id"] == "s0" else 3),
            ("OBS1", 6 if row["scene_id"] == "s0" else 3),
            ("OBS2PLUS", 3),
        ]:
            output.append(
                {
                    **{k: row[k] for k in ("variant", "seed", "scene_id", "role", "query_id")},
                    "region": region,
                    "pixel_count": count,
                    "total_valid": 9,
                    "absrel_sum": count * row["depth_absrel"],
                    "absrel_mean": row["depth_absrel"] if count else None,
                }
            )
    return output


def test_region_null_and_partition_contributions():
    from mcss.mechanism_pilot.geometry_carrier_statistics import analyze_regions

    data = fixture()
    regions = region_fixture(data)
    result = analyze_regions(regions, data[3], seeds=data[4]["seeds"], draws=100)
    assert result["groups"]["C0"]["OBS0"]["conditional_absrel"]["empty_mask_scene_count"] == 1
    assert result["gains"]["OBS2PLUS"]["mean"] == pytest.approx(0.01)
    assert sum(
        result["groups"]["C0"][r]["additive_error_contribution"]["mean"]
        for r in ("OBS0", "OBS1", "OBS2PLUS")
    ) == pytest.approx(0.4)
    regions[0]["pixel_count"] = 1
    with pytest.raises(ValueError):
        analyze_regions(regions, data[3], seeds=data[4]["seeds"], draws=100)


def script(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_raw_only_eight_figures_report_and_gzip_reproduction(tmp_path):
    import gzip

    data = fixture()
    values = {
        "scene_split": data[3],
        "training_contract": {"seeds": data[4]["seeds"]},
        "raw/dev_query_results": data[0],
        "raw/dev_context_results": data[1],
        "raw/dev_matched_results": data[2],
        "raw/dev_region_results": region_fixture(data),
        "raw/dev_static_matched_results": data[4]["static_matched_rows"],
        "audit/dev_state_use": data[4]["state_use_audit"],
        "training_curves": {"training": [], "dev": []},
        "cost_analysis": {"variants": {v: {"training_seconds": 1.0} for v in ("C0", "C1")}},
    }
    for name, value in values.items():
        path = tmp_path / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    cli = script("analyze_geometry_carrier.py")
    result = cli.run(tmp_path)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 8
    report = script("report_geometry_carrier.py")
    fields = report.run(tmp_path)
    assert not fields["FRESH_QUALIFICATION_OPENED"]
    assert not fields["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    assert fields["TEST_TIME_INPUT"] == "RGB+CAMERA"
    text = (tmp_path / "README.md").read_text()
    assert all("**" + str(i) + ". " in text for i in range(1, 15))
    for p in (tmp_path / "raw").glob("*.json"):
        compressed = p.with_suffix(".json.gz")
        compressed.write_bytes(gzip.compress(p.read_bytes(), mtime=0))
        p.unlink()
    result2 = cli.run(tmp_path, tmp_path / "reproduced")
    assert result == result2
    audit = script("audit_geometry_carrier_statistics.py")
    evidence = audit.audit(tmp_path)
    assert evidence["status"] == "PASS"
    assert evidence["scientific_json_count"] == 6
    assert evidence["all_scientific_json_byte_exact"]
    assert not evidence["media_or_checkpoint_loaded"]
    modified = tmp_path / "qualification_results.json"
    modified.write_text(modified.read_text() + "\n")
    with pytest.raises(RuntimeError, match="reproduction failed"):
        audit.audit(tmp_path)
    failed = json.loads((tmp_path / "audit/statistics_reproduction_audit.json").read_text())
    assert failed["status"] == "FAIL"


def test_single_dominant_scene_fails_even_when_all_loso_positive():
    data = fixture()
    for row in data[0]:
        if row["scene_id"] == "s0" and row["variant"] == "C1" and row["method"] == "direct":
            row["depth_absrel"] -= 0.2
    result = run(data)
    checks = result["qualification_results"]["surface_checks"]
    assert checks["every_leave_one_scene_out_gain_positive"]
    assert result["static_results"]["surface_gain"]["positive_top1_contribution"] > 0.5
    assert not checks["positive_top1_share_at_most_0_5"]
    assert result["qualification_results"]["SURFACE_TRAINING_STATUS"] == "PARTIAL"


def test_static_opacity_gate_uses_c1_anchor_direct_masks_not_surface_masks():
    data = fixture()
    for row in data[2]:
        if row["variant"] == "C1":
            row["normalized_depth_absrel"] = 0.4
    result = run(data)
    assert not result["state_use_results"]["opacity_artifact_gate"]
    assert result["state_use_results"]["static_opacity_artifact_gate"]
    assert result["qualification_results"]["SURFACE_TRAINING_STATUS"] == "PARTIAL"
    assert result["qualification_results"]["STATIC_DEV_STATUS"] == "SUPPORTED"
    data = fixture()
    for row in data[4]["static_matched_rows"]:
        if row["variant"] == "C1":
            row["anchor_normalized_depth_absrel"] = row["direct_normalized_depth_absrel"] - 0.05
    result = run(data)
    assert result["state_use_results"]["opacity_artifact_gate"]
    assert not result["state_use_results"]["static_opacity_artifact_gate"]
    assert result["qualification_results"]["SURFACE_TRAINING_STATUS"] == "SUPPORTED"
    assert result["qualification_results"]["STATIC_DEV_STATUS"] == "PARTIAL"


@pytest.mark.parametrize(
    "fault", ["missing", "imputed_zero", "null_value", "comparison", "denominator"]
)
def test_static_common_mask_rows_fail_closed(fault):
    data = fixture()
    rows = data[4]["static_matched_rows"]
    if fault == "missing":
        rows.pop()
    if fault == "imputed_zero":
        rows[0]["common_valid_count"] = 0
    if fault == "null_value":
        rows[0]["direct_depth_absrel"] = None
    if fault == "comparison":
        rows[0]["comparison"] = "direct_minus_anchor"
    if fault == "denominator":
        rows[0]["total_gt_valid"] = 10
    with pytest.raises(ValueError):
        run(data)


def test_secondary_c2_never_changes_primary_masks_or_gates():
    data = fixture()
    baseline = run(copy.deepcopy(data))
    for rows in (data[0], data[1], data[4]["static_matched_rows"]):
        clones = []
        for row in rows:
            if row["variant"] != "C1":
                continue
            clone = {**copy.deepcopy(row), "variant": "C2"}
            if clone.get("method", "direct") == "direct":
                for key in ("depth_absrel", "direct_depth_absrel"):
                    if key in clone:
                        clone[key] -= 0.02
            clones.append(clone)
        rows.extend(clones)
    result = run(data)
    assert result["static_results"]["primary"] == "C1_SURFACE16"
    assert result["static_results"]["secondary_variants"] == ["C2"]
    assert result["static_results"]["surface_gain"] == baseline["static_results"]["surface_gain"]
    assert result["qualification_results"] == baseline["qualification_results"]
    for key in ("common_mask_gains", "opacity_checks", "static_opacity_checks"):
        assert result["state_use_results"][key] == baseline["state_use_results"][key]
    assert result["static_results"]["free_space_extra_gain"]["mean"] == pytest.approx(0.02)
