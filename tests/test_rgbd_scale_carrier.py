"""V6 training scale: variant mapping, frozen TRAIN sets, CPU-only runner, report semantics."""

import importlib.util
import json
from pathlib import Path

import pytest

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.mechanism_pilot.rgbd_evidence_carrier import RGBDEvidenceCarrier
from mcss.mechanism_pilot.rgbd_scale_carrier import (
    VARIANT_SPECS,
    make_carrier,
    parameter_hash,
    train_records,
)


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variants_map_to_frozen_v5_recipes_with_shared_initialization():
    assert {v: s["train"] for v, s in VARIANT_SPECS.items()} == {
        "C0": "TRAIN24",
        "C1": "TRAIN72",
        "C2": "TRAIN72",
    }
    carriers = {v: make_carrier(v, 7, "cpu") for v in VARIANT_SPECS}
    assert isinstance(carriers["C0"], RGBDEvidenceCarrier)
    assert isinstance(carriers["C1"], RGBDEvidenceCarrier)
    assert type(carriers["C2"]) is DynamicSceneCarrier
    assert len({parameter_hash(c) for c in carriers.values()}) == 1
    counts = {v: sum(p.numel() for p in c.parameters()) for v, c in carriers.items()}
    assert counts == {"C0": 5150, "C1": 5150, "C2": 5125}


def test_train_records_follow_the_frozen_sets_exactly():
    scenes = [{"scene_id": f"s{i:02d}", "split": "TRAIN"} for i in range(72)]
    scenes += [{"scene_id": "d0", "split": "DEV"}]
    config = {
        "train_sets": {
            "TRAIN24": [f"s{i:02d}" for i in range(24)],
            "TRAIN72": [f"s{i:02d}" for i in range(72)],
        }
    }
    manifest = {"scenes": scenes}
    assert [r["scene_id"] for r in train_records(manifest, config, "C0")] == config["train_sets"][
        "TRAIN24"
    ]
    assert len(train_records(manifest, config, "C1")) == 72
    with pytest.raises(PermissionError):
        train_records({"scenes": scenes[:60]}, config, "C1")


def test_v6_runner_trains_only_v6_scripts_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_scale_experiment.py")
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
    assert [a[0] for a in seen] == ["scripts/train_rgbd_scale_carrier.py"] * 3
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)


def test_v6_report_uses_scale_semantics(tmp_path):
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
    script("analyze_rgbd_scale_carrier.py").run(tmp_path)
    report = script("report_rgbd_scale_carrier.py")
    fields = report.run(tmp_path)
    assert fields["SCALE_GAIN"] == "0.010000" and fields["SCALE_STATUS"] == "SUPPORTED"
    assert fields["INTERPRETATION_BRANCH"] == "B" and fields["DEVICE"] == "cpu"
    stats = {"mean": 0.1, "ci95": [0.05, 0.15]}
    cell = {"reference_label": "ABOVE", "vs_REF_TRAIN_ABSREL_OPTIMAL": stats}
    carrier = {
        r: {m: cell for m in ("depth_absrel", "depth_delta1")}
        for r in ("RAW", "OPACITY_NORMALIZED", "MEDIAN_SCALED")
    }
    reference = {
        "summary": {"CARRIER_REFERENCE_STATUS": "SOME_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE"},
        "carriers": {"V6_C0": carrier, "V6_C1": carrier},
        "sensitivity_excluding_no_hit_scenes": {
            "excluded_no_hit_scenes": ["x"],
            "carriers": {"V6_C0": carrier, "V6_C1": carrier},
        },
    }
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    fields = report.run(tmp_path)
    assert fields["INTERPRETATION_BRANCH"] == "A" and fields["C1_ABOVE_REFERENCE_HIT_SCENES"]
    assert all(f"**{i}. " in (tmp_path / "README.md").read_text() for i in range(1, 10))
