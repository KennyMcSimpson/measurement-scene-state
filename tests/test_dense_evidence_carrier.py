"""V3 sparse/dense evidence carrier: single-factor contract and V3-specific pipeline pieces."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_mechanism_training import observation

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import hash_scene_state
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import geometry_carrier
from mcss.mechanism_pilot.dense_evidence_carrier import (
    VARIANT_SPECS,
    build_state,
    make_carrier,
    parameter_hash,
    training_loss,
)
from mcss.types import Cameras


def context():
    torch.set_num_threads(1)
    observations = [observation(i) for i in (0, 1, 2)]
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    query = observation(5).camera
    cameras = Cameras(query.intrinsics[None, None], query.c2w[None, None], (8, 8))
    cameras = transform_cameras(cameras, torch.linalg.inv(observations[0].camera.c2w))
    return observations, bounds, cameras


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variants_share_parameters_and_differ_only_in_candidates():
    carriers = {v: make_carrier(v, 20260928, "cpu") for v in VARIANT_SPECS}
    assert len({parameter_hash(c) for c in carriers.values()}) == 1
    assert {v: c.config.token_count for v, c in carriers.items()} == {
        "C0": 128,
        "C1": 4096,
        "C2": 4096,
    }
    assert torch.equal(carriers["C1"]._candidate_ids, torch.arange(4096))
    assert all(sum(p.numel() for p in c.parameters()) == 5125 for c in carriers.values())
    assert parameter_hash(make_carrier("C0", 20260929, "cpu")) != parameter_hash(carriers["C0"])


def test_sparse_variant_reproduces_the_v2_state_exactly():
    observations, bounds, _ = context()
    v3 = make_carrier("C0", 7, "cpu")
    torch.manual_seed(7)
    v2 = DynamicSceneCarrier(CarrierConfig(grid_size=(16, 16, 16)))
    state3, _ = build_state(v3, observations, bounds, "same")
    state2, _ = geometry_carrier.build_state(v2, observations, bounds, "same")
    assert hash_scene_state(state3) == hash_scene_state(state2)


def test_only_frozen_candidate_counts_and_rgb_inputs_are_accepted():
    observations, bounds, cameras = context()
    torch.manual_seed(1)
    odd = DynamicSceneCarrier(CarrierConfig(grid_size=(16, 16, 16), token_count=256))
    with pytest.raises(ValueError):
        build_state(odd, observations, bounds, "odd")
    dense = make_carrier("C1", 1, "cpu")
    with pytest.raises(TypeError):
        build_state(dense, observations, bounds, "illegal", depth=torch.ones(1))
    state, _ = build_state(dense, observations, bounds, "dense")
    assert tuple(state.spatial_shape) == (16, 16, 16)
    assert torch.isfinite(state.density_logits).all()


def test_dense_and_sparse_states_differ_but_share_parameters():
    observations, bounds, _ = context()
    sparse, dense = make_carrier("C0", 3, "cpu"), make_carrier("C1", 3, "cpu")
    a, _ = build_state(sparse, observations, bounds, "a")
    b, _ = build_state(dense, observations, bounds, "b")
    assert parameter_hash(sparse) == parameter_hash(dense)
    assert not torch.equal(a.density_logits, b.density_logits)


def test_loss_mapping_uses_the_frozen_v2_terms():
    observations, bounds, cameras = context()
    state, _ = build_state(make_carrier("C1", 5, "cpu"), observations, bounds, "loss")
    rgb = torch.full((1, 1, 3, 8, 8), 0.6)
    depth = torch.full((1, 1, 1, 8, 8), 2.0)
    indices = torch.arange(64)
    v3 = {v: training_loss(state, cameras, rgb, depth, indices, v)[0] for v in VARIANT_SPECS}
    v2 = {
        v: geometry_carrier.training_loss(state, cameras, rgb, depth, indices, v)[0]
        for v in ("C0", "C1")
    }
    torch.testing.assert_close(v3["C0"], v2["C1"])
    torch.testing.assert_close(v3["C1"], v2["C1"])
    torch.testing.assert_close(v3["C2"], v2["C0"])


def test_v3_runner_trains_only_v3_scripts(tmp_path, monkeypatch):
    module = script("run_dense_evidence_experiment.py")
    seen = []
    monkeypatch.setattr(module, "execute", lambda root, label, args, **kwargs: seen.append(args[0]))
    config = {
        "seeds": [1],
        "variants": ["C0", "C1", "C2"],
        "primary_pair": ["C0", "C1"],
        "parallel_workers": 3,
        "infrastructure_retries": 2,
    }
    module.train_all(tmp_path, config, env=None, times={})
    assert seen == ["scripts/train_dense_evidence_carrier.py"] * 3


def test_v3_report_uses_density_semantics(tmp_path):
    from test_geometry_carrier_statistics import fixture, region_fixture

    data = fixture()
    values = {
        "scene_split": data[3],
        "training_contract": {"seeds": data[4]["seeds"], "steps": 3000},
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
    script("analyze_dense_evidence_carrier.py").run(tmp_path)
    fields = script("report_dense_evidence_carrier.py").run(tmp_path)
    assert fields["DENSITY_GAIN"] == "0.010000"
    assert fields["DENSITY_STATUS"] == "SUPPORTED"
    assert fields["SCENE_SPECIFICITY_STATUS"] == "SUPPORTED"
    assert fields["INTERPRETATION_BRANCH"] == "A"
    assert fields["FRESH_QUALIFICATION_INCLUDED"] is False
    assert not fields["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    text = (tmp_path / "README.md").read_text()
    assert "DENSITY_GAIN" in text and all(f"**{i}. " in text for i in range(1, 12))
