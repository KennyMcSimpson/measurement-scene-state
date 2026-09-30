"""FRESH-V2 qualification: independence audit, per-variant fresh bounds, seal before query GT."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_geometry_carrier_evaluation import fixture as evaluation_fixture

from mcss.mechanism_pilot.rgbd_bounds_carrier import make_carrier
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext
from mcss.mechanism_pilot.rgbd_fresh_v2_evaluation import (
    FreshDepthBoundsSceneData,
    FreshRGBDSceneData,
    FreshTrimmedDepthBoundsSceneData,
    evaluate_model,
    fresh_scene_data,
)


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_fixture(root):
    manifest, _ = evaluation_fixture(root)
    for record in manifest["scenes"]:
        if record["split"] == "DEV":
            record.update(split="FRESH_QUALIFICATION", historically_exposed=False)
            record["independence_verified"] = True
    manifest["fresh_status"] = "VERIFIED"
    return manifest


def test_independence_audit_accepts_clean_and_rejects_every_overlap():
    audit = script("prepare_rgbd_fresh_v2_qualification.py").independence_audit
    v7 = {"scenes": [{"scene_id": "ai_001_001", "physical_scene_id": "p1"}]}
    names = ("ai_001_002", "ai_002_001", "ai_002_002", "ai_002_003", "ai_003_001", "ai_004_001")
    partitions = [
        {
            "scene_name": s,
            "official_split": "train",
            "protocol_partition": "train",
            "previously_observed": "False",
        }
        for s in names
    ]

    def record(sid, physical):
        return {
            "scene_id": sid,
            "split": "FRESH_QUALIFICATION",
            "physical_scene_id": physical,
            "historically_exposed": False,
        }

    clean = [record("ai_002_001", "p2"), record("ai_002_002", "p3"), record("ai_003_001", "p4")]
    assert all(a["verified"] for a in audit(clean, v7, set(), set(), partitions, set()))
    broken = [
        ([record("ai_002_001", "p1")], set(), set()),  # physical identity shared with V7
        (clean, set(), {"p3"}),  # physical identity shared with an earlier cohort
        (clean, {"ai_003_001"}, set()),  # used or considered by an earlier experiment
        (clean + [record("ai_002_003", "p5")], set(), set()),  # three scenes in one volume
        ([record("ai_001_002", "p6")], set(), set()),  # volume shared with V7 TRAIN/DEV
        ([{**record("ai_002_001", "p2"), "historically_exposed": True}], set(), set()),
    ]
    for records, earlier, physical in broken:
        with pytest.raises(PermissionError):
            audit(records, v7, earlier, physical, partitions, set())
    with pytest.raises(PermissionError):
        audit(clean, v7, set(), set(), partitions, {"ai_003_001"})
    for key, value in (("previously_observed", "True"), ("official_split", "val")):
        changed = copy.deepcopy(partitions)
        changed[1][key] = value
        with pytest.raises(PermissionError):
            audit(clean, v7, set(), set(), changed, set())
    with pytest.raises(PermissionError):
        audit([record("ai_004_001", "p7")], v7, set(), set(), partitions[:-1], set())


def test_fresh_loaders_follow_each_v7_bounds_rule_and_never_serve_targets(tmp_path):
    manifest = fresh_fixture(tmp_path)
    record = next(r for r in manifest["scenes"] if r["split"] == "FRESH_QUALIFICATION")
    assert fresh_scene_data("C0") is FreshRGBDSceneData
    assert fresh_scene_data("C1") is FreshDepthBoundsSceneData
    assert fresh_scene_data("C2") is FreshTrimmedDepthBoundsSceneData
    frozen = FreshRGBDSceneData(record, manifest, tmp_path, "cpu", [])
    access = []
    depth = FreshDepthBoundsSceneData(record, manifest, tmp_path, "cpu", access)
    context = depth.context("A")
    assert isinstance(context, RGBDContext) and context.depths.shape[0] == len(context)
    assert not torch.equal(frozen.bounds["A"], depth.bounds["A"])
    assert all(
        e["purpose"] == "CONTEXT_ONLY_RGBD" for e in access if "depth" in e.get("channels", [])
    )
    with pytest.raises(PermissionError):
        depth.target(0)
    with pytest.raises(PermissionError):
        blocked = {**manifest, "fresh_status": "BLOCKED"}
        FreshDepthBoundsSceneData(record, blocked, tmp_path, "cpu", [])
    train = next(r for r in manifest["scenes"] if r["split"] == "TRAIN")
    with pytest.raises(PermissionError):
        FreshTrimmedDepthBoundsSceneData(train, manifest, tmp_path, "cpu", [])


def test_fresh_v2_evaluation_seals_every_state_before_query_depth(tmp_path):
    torch.set_num_threads(1)
    manifest = fresh_fixture(tmp_path)
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    boxes = {}
    for variant in ("C0", "C1"):
        carrier = make_carrier(variant, 5, "cpu")
        with torch.no_grad():
            carrier.depth_weight.fill_(0.05)
        out = tmp_path / f"evaluation_{variant}"
        result = evaluate_model(
            carrier, manifest, tmp_path, out, variant, 5, 100, "cpu", diagnostics=True
        )
        assert len(result["query_rows"]) == 40
        access = json.loads((out / "GT_access.json").read_text())
        marker = next(
            i for i, e in enumerate(access) if e.get("event") == "ALL_FRESH_STATES_SEALED"
        )
        early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
        assert early
        for event in early:
            roles = records[event["scene_id"]]["roles"]
            assert event["purpose"] == "CONTEXT_ONLY_RGBD"
            assert event["frame_id"] in set(roles["context_a"]) | set(roles["context_b"])
        metadata = json.loads((out / "state_hashes.json").read_text())
        boxes[variant] = {k: v["bounds"] for k, v in metadata.items()}
    assert any(boxes["C0"][k] != boxes["C1"][k] for k in boxes["C0"])
    dev = copy.deepcopy(manifest)
    for r in dev["scenes"]:
        if r["split"] == "FRESH_QUALIFICATION":
            r["split"] = "DEV"
    with pytest.raises(PermissionError):
        evaluate_model(carrier, dev, tmp_path, tmp_path / "blocked", "C1", 5, 100, "cpu")


def test_volume_bootstrap_resamples_volumes_not_scenes():
    report = script("report_rgbd_fresh_v2_qualification.py")
    result = report.volume_bootstrap(
        {"ai_001_001": 1.0, "ai_001_002": 3.0, "ai_002_001": -1.0}, draws=2000
    )
    assert result["n_volumes"] == 2 and result["volumes_positive"] == 1
    assert result["mean"] == pytest.approx(0.5)
    assert result["ci95"][0] == pytest.approx(-1.0) and result["ci95"][1] == pytest.approx(2.0)


def test_report_requires_all_four_gates(tmp_path):
    from test_geometry_carrier_statistics import fixture, region_fixture

    data = fixture()
    v7 = tmp_path / "v7"
    v7.mkdir()
    (v7 / "terminal_summary.txt").write_text("BOUNDS_GAIN=0.1\nINTERPRETATION_BRANCH=A\n")
    values = {
        "scene_split": data[3],
        "qualification_contract": {"seeds": data[4]["seeds"], "v7_root": str(v7)},
        "raw/fresh_query_results": data[0],
        "raw/fresh_context_results": data[1],
        "raw/fresh_matched_results": data[2],
        "raw/fresh_region_results": region_fixture(data),
        "raw/fresh_static_matched_results": data[4]["static_matched_rows"],
        "audit/fresh_state_use": data[4]["state_use_audit"],
        "training_curves": {"training": [], "dev": []},
        "cost_analysis": {"variants": {v: {"training_seconds": 1.0} for v in ("C0", "C1")}},
    }
    for name, value in values.items():
        path = tmp_path / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    artifacts = script("analyze_rgbd_fresh_v2_qualification.py").run(tmp_path)
    assert artifacts["ray_hit_fractions"]["descriptive_only"]
    scenes = sorted({r["scene_id"] for r in data[0]})

    def cell(label):
        return {
            "reference_label": label,
            "vs_REF_TRAIN_ABSREL_OPTIMAL": {
                "mean": 0.1,
                "ci95": [0.05, 0.15],
                "improved_tied_worse": [len(scenes), 0, 0],
                "per_scene": {s: 0.1 for s in scenes},
            },
        }

    report = script("report_rgbd_fresh_v2_qualification.py")
    for label in ("NOT_DISTINGUISHABLE", "ABOVE"):
        reference = {
            "summary": {"CARRIER_REFERENCE_STATUS": "X"},
            "carriers": {"V7_C1": {"RAW": {"depth_absrel": cell(label)}}},
        }
        (tmp_path / "reference_results.json").write_text(json.dumps(reference))
        fields = report.run(tmp_path)
        others = (
            fields["BOUNDS_STATUS_FRESH"],
            fields["SCENE_SPECIFICITY_STATUS_FRESH"],
            fields["STATIC_STATUS_FRESH"],
        )
        expected = label == "ABOVE" and all(g == "SUPPORTED" for g in others)
        assert fields["REFERENCE_STATUS_FRESH"] == (
            "SUPPORTED" if label == "ABOVE" else "NOT_ESTABLISHED"
        )
        assert (fields["FRESH_QUALIFICATION_STATUS"] == "QUALIFIED") == expected
        assert fields["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"] == expected
