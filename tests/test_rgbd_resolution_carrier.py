"""V8 voxel resolution: the grid is the only difference, and 16^3 reproduces V5/V7 exactly."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_geometry_carrier_evaluation import fixture as evaluation_fixture
from test_mechanism_training import observation

from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import geometry_carrier
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.rgbd_bounds_carrier import SCENE_DATA
from mcss.mechanism_pilot.rgbd_resolution_carrier import (
    BOUNDS_RULE,
    GRIDS,
    VARIANT_SPECS,
    ResolutionRGBDCarrier,
    build_state,
    load_checkpoint,
    make_carrier,
    parameter_hash,
    save_checkpoint,
    scene_data,
    training_loss,
)
from mcss.mechanism_pilot.rgbd_resolution_evaluation import evaluate_model
from mcss.types import Cameras


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context():
    torch.set_num_threads(1)
    observations = [observation(i) for i in (0, 1, 2)]
    depths = torch.full((3, 8, 8), 2.0)
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    query = observation(5).camera
    cameras = Cameras(query.intrinsics[None, None], query.c2w[None, None], (8, 8))
    cameras = transform_cameras(cameras, torch.linalg.inv(observations[0].camera.c2w))
    return v5.RGBDContext(observations, depths), bounds, cameras


def targets():
    return torch.full((1, 1, 3, 8, 8), 0.6), torch.full((1, 1, 1, 8, 8), 2.0), torch.arange(64)


def test_the_grid_is_the_only_difference_between_variants():
    assert {v: s["grid"] for v, s in VARIANT_SPECS.items()} == {"C0": 16, "C1": 32, "C2": 24}
    assert {s["model"] for s in VARIANT_SPECS.values()} == {"C1"}
    assert {s["train"] for s in VARIANT_SPECS.values()} == {"TRAIN72"}
    assert set(GRIDS) == {16, 24, 32}
    assert {scene_data(v) for v in VARIANT_SPECS} == {SCENE_DATA[BOUNDS_RULE]}
    carriers = {v: make_carrier(v, 3, "cpu") for v in VARIANT_SPECS}
    assert len({parameter_hash(c) for c in carriers.values()}) == 1
    assert parameter_hash(carriers["C0"]) == parameter_hash(v5.make_carrier("C1", 3, "cpu"))
    for v, c in carriers.items():
        assert sum(p.numel() for p in c.parameters()) == 5150
        assert c._candidate_points.shape[0] == VARIANT_SPECS[v]["grid"] ** 3


def test_grid16_reproduces_the_v5_state_and_loss_exactly():
    rgbd, bounds, cameras = context()
    old = v5.make_carrier("C1", 11, "cpu")
    new = make_carrier("C0", 11, "cpu")
    assert hash_value(old.state_dict()) == hash_value(new.state_dict())
    with torch.no_grad():
        for carrier in (old, new):
            carrier.depth_weight.fill_(0.2)
    first, _ = v5.build_state(old, rgbd, bounds, "same")
    second, _ = build_state(new, rgbd, bounds, "same")
    assert hash_scene_state(first) == hash_scene_state(second)
    rgb, depth, indices = targets()
    a, a_terms = geometry_carrier.training_loss(first, cameras, rgb, depth, indices, "C1")
    b, b_terms = training_loss(second, cameras, rgb, depth, indices, "C0")
    assert torch.equal(a, b)
    assert all(
        torch.equal(torch.as_tensor(a_terms[k]), torch.as_tensor(b_terms[k])) for k in a_terms
    )


@pytest.mark.parametrize("variant", ["C1", "C2"])
def test_finer_grids_build_train_and_roundtrip(tmp_path, variant):
    rgbd, bounds, cameras = context()
    carrier = make_carrier(variant, 5, "cpu")
    state, _ = build_state(carrier, rgbd, bounds, "fine")
    g = VARIANT_SPECS[variant]["grid"]
    assert tuple(state.spatial_shape) == (g, g, g)
    loss, _ = training_loss(state, cameras, *targets(), variant)
    loss.backward()
    grad = carrier.depth_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    path = tmp_path / "carrier.pt"
    save_checkpoint(path, carrier, variant=variant, seed=5, step=1, lock_sha256="x")
    restored, payload = load_checkpoint(path, "cpu")
    assert isinstance(restored, ResolutionRGBDCarrier)
    assert payload["config"]["grid_size"] in ([g, g, g], (g, g, g))
    with pytest.raises(PermissionError):
        v5.load_checkpoint(path, "cpu")


def test_unregistered_grids_and_contexts_are_rejected():
    rgbd, bounds, cameras = context()
    carrier = make_carrier("C0", 5, "cpu")
    with pytest.raises(TypeError):
        build_state(carrier, tuple(rgbd), bounds, "plain")
    with pytest.raises(TypeError):
        build_state(v5.make_carrier("C1", 5, "cpu"), rgbd, bounds, "v5")
    coarse = ResolutionRGBDCarrier(CarrierConfig(grid_size=(8, 8, 8), token_count=512))
    with pytest.raises(ValueError):
        build_state(coarse, rgbd, bounds, "coarse")
    state, _ = build_state(carrier, rgbd, bounds, "ok")
    with pytest.raises(KeyError):
        training_loss(state, cameras, *targets(), "C9")


def test_evaluation_seals_states_and_shuffles_every_voxel(tmp_path):
    torch.set_num_threads(1)
    manifest, _ = evaluation_fixture(tmp_path)
    for variant in ("C0", "C1"):
        carrier = make_carrier(variant, 5, "cpu")
        out = tmp_path / f"evaluation_{variant}"
        evaluate_model(carrier, manifest, tmp_path, out, variant, 5, 100, "cpu", diagnostics=True)
        access = json.loads((out / "GT_access.json").read_text())
        marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_DEV_STATES_SEALED")
        assert all(
            e["purpose"] == "CONTEXT_ONLY_RGBD"
            for e in access[:marker]
            if "depth" in e.get("channels", [])
        )
        rows = json.loads((out / "query_results.json").read_text())
        assert {r["method"] for r in rows} >= {"direct", "spatial_shuffle", "wrong_scene", "zero"}


def test_v8_runner_trains_only_v8_scripts_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_resolution_experiment.py")
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
    assert [a[0] for a in seen] == ["scripts/train_rgbd_resolution_carrier.py"] * 3
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)


def test_v8_report_requires_the_all_scene_reference_for_branch_a(tmp_path):
    from test_geometry_carrier_statistics import fixture, region_fixture

    data = fixture()
    values = {
        "scene_split": data[3],
        "training_contract": {"seeds": data[4]["seeds"], "steps": 6000, "bounds_rule": "X"},
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
    script("analyze_rgbd_resolution_carrier.py").run(tmp_path)
    report = script("report_rgbd_resolution_carrier.py")
    fields = report.run(tmp_path)
    assert fields["RESOLUTION_GAIN"] == "0.010000" and fields["RESOLUTION_STATUS"] == "SUPPORTED"
    assert fields["INTERPRETATION_BRANCH"] == "B" and fields["BOUNDS_RULE"] == "X"
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
        "carriers": {"V8_C0": carrier(below), "V8_C1": carrier(below)},
        "sensitivity_excluding_no_hit_scenes": {
            "excluded_no_hit_scenes": ["x"],
            "carriers": {"V8_C0": carrier(above), "V8_C1": carrier(above)},
        },
    }
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    assert report.run(tmp_path)["INTERPRETATION_BRANCH"] == "B"
    reference["carriers"]["V8_C1"] = carrier(above)
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    fields = report.run(tmp_path)
    assert fields["INTERPRETATION_BRANCH"] == "A" and fields["C1_ABOVE_REFERENCE_ALL_SCENES"]
