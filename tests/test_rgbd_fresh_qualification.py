"""FRESH-V1 qualification: independence audit, fresh loader guards, seal-before-query-GT."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_geometry_carrier_evaluation import fixture as evaluation_fixture

from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDContext, make_carrier
from mcss.mechanism_pilot.rgbd_fresh_evaluation import FreshRGBDSceneData, evaluate_model


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
    audit = script("prepare_rgbd_fresh_qualification.py").independence_audit
    v5 = {"scenes": [{"scene_id": "ai_001_001", "physical_scene_id": "p1"}]}
    partitions = [
        {
            "scene_name": "ai_002_001",
            "official_split": "val",
            "protocol_partition": "val",
            "previously_observed": "False",
        },
        {
            "scene_name": "ai_003_001",
            "official_split": "val",
            "protocol_partition": "val",
            "previously_observed": "False",
        },
    ]

    def record(sid, physical):
        return {
            "scene_id": sid,
            "split": "FRESH_QUALIFICATION",
            "physical_scene_id": physical,
            "historically_exposed": False,
        }

    clean = [record("ai_002_001", "p2"), record("ai_003_001", "p3")]
    assert all(a["verified"] for a in audit(clean, v5, partitions, set()))
    broken = [
        [record("ai_002_001", "p1")],  # physical identity shared with V5
        [record("ai_002_001", "p2"), record("ai_002_001", "p9")],  # same volume twice
        [{**record("ai_002_001", "p2"), "historically_exposed": True}],
    ]
    for records in broken:
        with pytest.raises(PermissionError):
            audit(records, v5, partitions, set())
    with pytest.raises(PermissionError):
        audit(clean, v5, partitions, {"ai_003_001"})
    observed = copy.deepcopy(partitions)
    observed[0]["previously_observed"] = "True"
    with pytest.raises(PermissionError):
        audit(clean, v5, observed, set())
    with pytest.raises(PermissionError):
        audit([record("ai_001_002", "p8")], v5, partitions, set())  # V5 volume, not val


def test_fresh_loader_accepts_only_verified_fresh_and_never_serves_targets(tmp_path):
    manifest = fresh_fixture(tmp_path)
    record = next(r for r in manifest["scenes"] if r["split"] == "FRESH_QUALIFICATION")
    data = FreshRGBDSceneData(record, manifest, tmp_path, "cpu", [])
    context = data.context("A")
    assert isinstance(context, RGBDContext) and context.depths.shape[0] == len(context)
    with pytest.raises(PermissionError):
        data.target(0)
    with pytest.raises(PermissionError):
        FreshRGBDSceneData(record, {**manifest, "fresh_status": "BLOCKED"}, tmp_path, "cpu", [])
    train = next(r for r in manifest["scenes"] if r["split"] == "TRAIN")
    with pytest.raises(PermissionError):
        FreshRGBDSceneData(train, manifest, tmp_path, "cpu", [])


def test_fresh_evaluation_reads_query_depth_only_after_every_fresh_state_is_sealed(tmp_path):
    torch.set_num_threads(1)
    manifest = fresh_fixture(tmp_path)
    carrier = make_carrier("C1", 5, "cpu")
    with torch.no_grad():
        carrier.depth_weight.fill_(0.05)
    out = tmp_path / "evaluation"
    result = evaluate_model(carrier, manifest, tmp_path, out, "C1", 5, 100, "cpu", diagnostics=True)
    assert len(result["query_rows"]) == 40
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_FRESH_STATES_SEALED")
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert early
    for event in early:
        roles = records[event["scene_id"]]["roles"]
        assert event["purpose"] == "CONTEXT_ONLY_RGBD"
        assert event["frame_id"] in set(roles["context_a"]) | set(roles["context_b"])
    assert all(e.get("scene_id", "") != "train_unused" for e in access)
    with pytest.raises(PermissionError):
        dev = copy.deepcopy(manifest)
        for r in dev["scenes"]:
            if r["split"] == "FRESH_QUALIFICATION":
                r["split"] = "DEV"
        evaluate_model(carrier, dev, tmp_path, tmp_path / "blocked", "C1", 5, 100, "cpu")
