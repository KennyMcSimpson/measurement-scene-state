"""V9 training view count: only the context view count differs; 3 views reproduce V8 exactly."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_mechanism_training import observation

from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot.rgbd_viewcount_carrier import (
    EXTRA_MAX,
    GRID,
    VARIANT_SPECS,
    build_state,
    extra_count,
    make_carrier,
    parameter_hash,
    with_extra_views,
)


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contexts():
    torch.set_num_threads(1)
    observations = [observation(i) for i in range(7)]
    depths = torch.full((7, 8, 8), 2.0)
    context = v5.RGBDContext([observations[i] for i in (0, 4, 5)], depths[:3])
    stream = v5.RGBDContext([observations[i] for i in (1, 2, 3, 6)], depths[3:])
    return context, stream, torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])


def test_variants_share_the_model_and_differ_only_by_extra_views():
    assert GRID in (16, 32)
    carriers = [make_carrier(v, 3, "cpu") for v in VARIANT_SPECS]
    assert len({parameter_hash(c) for c in carriers}) == 1
    assert {sum(p.numel() for p in c.parameters()) for c in carriers} == {5150}
    assert {tuple(c.config.grid_size) for c in carriers} == {(GRID, GRID, GRID)}
    generator = torch.Generator().manual_seed(1)
    assert {extra_count("C0", generator) for _ in range(20)} == {0}
    assert {extra_count("C2", generator) for _ in range(20)} == {EXTRA_MAX}
    draws = {extra_count("C1", generator) for _ in range(200)}
    assert draws == set(range(EXTRA_MAX + 1))
    untouched = torch.Generator().manual_seed(5)
    before = untouched.get_state()
    extra_count("C0", untouched)
    extra_count("C2", untouched)
    assert torch.equal(untouched.get_state(), before)


def test_extra_views_are_appended_in_arrival_order_and_zero_reproduces_v8():
    context, stream, bounds = contexts()
    assert with_extra_views(context, stream, 0) is context
    extended = with_extra_views(context, stream, 3)
    assert [o.frame_id for o in extended] == list(range(6))
    assert torch.equal(extended.depths, torch.cat((context.depths, stream.depths[:3])))
    with pytest.raises(ValueError):
        with_extra_views(context, stream, 5)
    carrier = make_carrier("C1", 4, "cpu")
    with torch.no_grad():
        carrier.depth_weight.fill_(0.05)
    reference = v8.ResolutionRGBDCarrier(carrier.config)
    reference.load_state_dict(carrier.state_dict())
    expected, _ = v8.build_state(reference, context, bounds, "x")
    state, _ = build_state(carrier, with_extra_views(context, stream, 0), bounds, "x")
    assert hash_scene_state(state) == hash_scene_state(expected)
    more, _ = build_state(carrier, extended, bounds, "x")
    assert hash_scene_state(more) != hash_scene_state(expected)


def test_v9_runner_trains_only_v9_scripts_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_viewcount_experiment.py")
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
    assert [a[0] for a in seen] == ["scripts/train_rgbd_viewcount_carrier.py"] * 3
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)


def test_v9_report_requires_the_all_scene_reference_for_branch_a(tmp_path):
    from test_geometry_carrier_statistics import fixture, region_fixture

    data = fixture()
    values = {
        "scene_split": data[3],
        "training_contract": {"seeds": data[4]["seeds"], "steps": 6000, "grid": GRID},
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
    script("analyze_rgbd_viewcount_carrier.py").run(tmp_path)
    report = script("report_rgbd_viewcount_carrier.py")
    fields = report.run(tmp_path)
    assert fields["VIEWCOUNT_GAIN"] == "0.010000" and fields["VIEWCOUNT_STATUS"] == "SUPPORTED"
    assert fields["INTERPRETATION_BRANCH"] == "B" and fields["GRID"] == GRID
    stats = {"mean": 0.1, "ci95": [0.05, 0.15]}

    def carrier(label):
        cell = {"reference_label": label, "vs_REF_TRAIN_ABSREL_OPTIMAL": stats}
        return {
            r: {m: cell for m in ("depth_absrel", "depth_delta1")}
            for r in ("RAW", "OPACITY_NORMALIZED", "MEDIAN_SCALED")
        }

    reference = {
        "summary": {"CARRIER_REFERENCE_STATUS": "SOME_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE"},
        "carriers": {"V9_C0": carrier("NOT_DISTINGUISHABLE"), "V9_C1": carrier("ABOVE")},
    }
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    fields = report.run(tmp_path)
    assert fields["INTERPRETATION_BRANCH"] == "A" and fields["C1_ABOVE_REFERENCE_ALL_SCENES"]
