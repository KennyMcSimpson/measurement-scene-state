"""V7 bounds rule: identical recipe, per-variant bounds in training and evaluation, CPU runner."""

import importlib.util
import json
from pathlib import Path

import torch
from test_geometry_carrier_evaluation import fixture as evaluation_fixture

from mcss.mechanism_pilot.rgbd_bounds_carrier import (
    VARIANT_SPECS,
    make_carrier,
    parameter_hash,
    scene_data,
)
from mcss.mechanism_pilot.rgbd_bounds_evaluation import evaluate_model
from mcss.mechanism_pilot.rgbd_depth_bounds import (
    DepthBoundsSceneData,
    TrimmedDepthBoundsSceneData,
)
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDEvidenceCarrier, RGBDSceneData


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variants_differ_only_by_their_bounds_rule():
    assert {s["model"] for s in VARIANT_SPECS.values()} == {"C1"}
    assert {s["train"] for s in VARIANT_SPECS.values()} == {"TRAIN72"}
    assert scene_data("C0") is RGBDSceneData
    assert scene_data("C1") is DepthBoundsSceneData
    assert scene_data("C2") is TrimmedDepthBoundsSceneData
    carriers = [make_carrier(v, 3, "cpu") for v in VARIANT_SPECS]
    assert all(isinstance(c, RGBDEvidenceCarrier) for c in carriers)
    assert len({parameter_hash(c) for c in carriers}) == 1
    assert {sum(p.numel() for p in c.parameters()) for c in carriers} == {5150}


def test_evaluation_builds_states_with_the_variants_own_bounds(tmp_path):
    torch.set_num_threads(1)
    manifest, _ = evaluation_fixture(tmp_path)
    carrier = make_carrier("C0", 5, "cpu")
    boxes = {}
    for variant in ("C0", "C1"):
        out = tmp_path / f"evaluation_{variant}"
        evaluate_model(carrier, manifest, tmp_path, out, variant, 5, 100, "cpu")
        metadata = json.loads((out / "state_hashes.json").read_text())
        boxes[variant] = {k: v["bounds"] for k, v in metadata.items()}
        access = json.loads((out / "GT_access.json").read_text())
        marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_DEV_STATES_SEALED")
        assert all(
            e["purpose"] == "CONTEXT_ONLY_RGBD"
            for e in access[:marker]
            if "depth" in e.get("channels", [])
        )
    assert boxes["C0"].keys() == boxes["C1"].keys()
    assert any(boxes["C0"][k] != boxes["C1"][k] for k in boxes["C0"])
    for key, box in boxes["C1"].items():
        anchor_key = key.replace("anchor_", "")
        assert box == boxes["C1"][anchor_key]  # anchor states reuse the full-role box


def test_v7_runner_trains_only_v7_scripts_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_bounds_experiment.py")
    seen = []
    monkeypatch.setattr(module, "execute", lambda root, label, args, **kwargs: seen.append(args))
    config = {
        "seeds": [1],
        "variants": ["C0", "C1", "C2"],
        "primary_pair": ["C0", "C1"],
        "parallel_workers": 3,
        "infrastructure_retries": 2,
    }
    module.train_all(tmp_path, config, env=None, times={})
    assert [a[0] for a in seen] == ["scripts/train_rgbd_bounds_carrier.py"] * 3
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)


def test_v7_report_requires_the_all_scene_reference_for_branch_a(tmp_path):
    from test_geometry_carrier_statistics import fixture, region_fixture

    data = fixture()
    values = {
        "scene_split": data[3],
        "training_contract": {"seeds": data[4]["seeds"], "steps": 6000},
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
    script("analyze_rgbd_bounds_carrier.py").run(tmp_path)
    report = script("report_rgbd_bounds_carrier.py")
    fields = report.run(tmp_path)
    assert fields["BOUNDS_GAIN"] == "0.010000" and fields["BOUNDS_STATUS"] == "SUPPORTED"
    assert fields["INTERPRETATION_BRANCH"] == "B"
    stats = {"mean": 0.1, "ci95": [0.05, 0.15]}
    above = {"reference_label": "ABOVE", "vs_REF_TRAIN_ABSREL_OPTIMAL": stats}
    below = {"reference_label": "NOT_DISTINGUISHABLE", "vs_REF_TRAIN_ABSREL_OPTIMAL": stats}

    def carrier(cell):
        return {
            r: {m: cell for m in ("depth_absrel", "depth_delta1")}
            for r in ("RAW", "OPACITY_NORMALIZED", "MEDIAN_SCALED")
        }

    reference = {
        "summary": {"CARRIER_REFERENCE_STATUS": "SOME_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE"},
        "carriers": {"V7_C0": carrier(below), "V7_C1": carrier(below)},
        "sensitivity_excluding_no_hit_scenes": {
            "excluded_no_hit_scenes": ["x"],
            "carriers": {"V7_C0": carrier(above), "V7_C1": carrier(above)},
        },
    }
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    assert report.run(tmp_path)["INTERPRETATION_BRANCH"] == "B"  # hit-scene ABOVE only
    reference["carriers"]["V7_C1"] = carrier(above)
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    fields = report.run(tmp_path)
    assert fields["INTERPRETATION_BRANCH"] == "A" and fields["C1_ABOVE_REFERENCE_ALL_SCENES"]
