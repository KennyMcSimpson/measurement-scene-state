"""V11 carrier width: only the width differs; width 8 reproduces V9; EVAL-V3 gates and runner."""

import importlib.util
from pathlib import Path

import pytest
import torch
from test_mechanism_training import observation

from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot import rgbd_viewcount_carrier as v9
from mcss.mechanism_pilot.rgbd_completion_carrier import (
    BASE_EXTRA,
    VARIANT_SPECS,
    build_state,
    extra_count,
    load_checkpoint,
    make_carrier,
    save_checkpoint,
    training_loss,
)
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
    rgbd = v5.RGBDContext(observations, torch.full((3, 8, 8), 2.0))
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    query = observation(5).camera
    cameras = Cameras(query.intrinsics[None, None], query.c2w[None, None], (8, 8))
    cameras = transform_cameras(cameras, torch.linalg.inv(observations[0].camera.c2w))
    return rgbd, bounds, cameras


def targets():
    return torch.full((1, 1, 3, 8, 8), 0.6), torch.full((1, 1, 1, 8, 8), 2.0), torch.arange(64)


def test_only_the_width_differs_and_width8_is_the_v9_draw():
    assert {v: (s["hidden"], s["expansion"]) for v, s in VARIANT_SPECS.items()} == {
        "C0": (8, 16),
        "C1": (32, 64),
    }
    assert {s["train"] for s in VARIANT_SPECS.values()} == {"TRAIN72"}
    assert BASE_EXTRA == {"C0": "none", "C1": "uniform_0_4"}
    narrow, wide = make_carrier("C0", 3, "cpu"), make_carrier("C1", 3, "cpu")
    assert hash_value(narrow.state_dict()) == hash_value(
        v9.make_carrier("C0", 3, "cpu").state_dict()
    )
    assert sum(p.numel() for p in narrow.parameters()) == 5150
    assert sum(p.numel() for p in wide.parameters()) == 64238
    assert wide._candidate_points.shape[0] == 32**3


def test_width8_reproduces_the_v8_state_and_loss_exactly():
    rgbd, bounds, cameras = context()
    old, new = v8.make_carrier("C1", 11, "cpu"), make_carrier("C0", 11, "cpu")
    with torch.no_grad():
        for carrier in (old, new):
            carrier.depth_weight.fill_(0.2)
    first, _ = v8.build_state(old, rgbd, bounds, "same")
    second, _ = build_state(new, rgbd, bounds, "same")
    assert hash_scene_state(first) == hash_scene_state(second)
    a, _ = v8.training_loss(first, cameras, *targets(), "C0")
    b, _ = training_loss(second, cameras, *targets(), "C0")
    assert torch.equal(a, b)


def test_extra_view_rules_consume_the_generator_like_v9():
    for base, v9_variant in (("none", "C0"), ("uniform_0_4", "C1")):
        mine, theirs = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
        drawn = [extra_count(base, mine) for _ in range(20)]
        assert drawn == [v9.extra_count(v9_variant, theirs) for _ in range(20)]
    assert set(drawn) <= set(range(5)) and len(set(drawn)) > 1
    with pytest.raises(ValueError):
        extra_count("all_4", torch.Generator())


def test_the_wide_carrier_trains_and_roundtrips(tmp_path):
    rgbd, bounds, cameras = context()
    carrier = make_carrier("C1", 5, "cpu")
    state, _ = build_state(carrier, rgbd, bounds, "wide")
    assert tuple(state.spatial_shape) == (32, 32, 32)
    loss, _ = training_loss(state, cameras, *targets(), "C1")
    loss.backward()
    assert carrier.refinement[0].weight.grad.abs().sum() > 0
    path = tmp_path / "wide.pt"
    save_checkpoint(path, carrier, variant="C1", seed=5, step=1, lock_sha256="x")
    restored, payload = load_checkpoint(path, "cpu")
    assert payload["config"]["hidden_dim"] == 32
    assert hash_value(restored.state_dict()) == hash_value(carrier.state_dict())
    with pytest.raises(PermissionError):
        v8.load_checkpoint(path, "cpu")
    odd = v8.ResolutionRGBDCarrier(
        CarrierConfig(grid_size=(32, 32, 32), token_count=32**3, hidden_dim=16, expansion_dim=32)
    )
    with pytest.raises(ValueError):
        build_state(odd, rgbd, bounds, "odd")


def make_row(base, noise, method, seed, far, total):
    return {
        **base,
        "method": method,
        "seed": seed,
        "absrel_ALL": total + noise,
        "absrel_NEAR": 0.2,
        "absrel_FAR": far + noise,
        "depth_absrel": total + noise,
    }


def rows_for(c0_far, c1_far, reproj_all, fill_all, scenes=8):
    rows = []
    for i in range(scenes):
        for role in ("A", "B"):
            for qid in (8, 9):
                noise = 0.002 * ((i + qid) % 3)
                base = {"scene_id": f"s{i}", "role": role, "query_id": qid, "far_fraction": 0.2}
                rows.append(make_row(base, noise, "REPROJ_NN", 0, 0.46, reproj_all))
                for seed in (1, 2):
                    rows.append(make_row(base, noise, "C0", seed, c0_far, 0.33))
                    rows.append(make_row(base, noise, "C1", seed, c1_far, 0.33))
                    rows.append(make_row(base, noise, "FILL8_C0", seed, c0_far, fill_all + 0.01))
                    rows.append(make_row(base, noise, "FILL8_C1", seed, c1_far, fill_all))
    return rows


def test_eval_v3_gates_set_the_branch():
    analyze_rows = script("evaluate_rgbd_completion_eval_v3.py").analyze_rows
    result = analyze_rows(rows_for(0.41, 0.35, 0.29, 0.26))
    assert result["COMPLETION_GAIN"]["mean"] == pytest.approx(0.06)
    assert result["statuses"] == {"COMPLETION_STATUS": "SUPPORTED", "HYBRID_STATUS": "SUPPORTED"}
    assert result["INTERPRETATION_BRANCH"] == "A"
    result = analyze_rows(rows_for(0.41, 0.35, 0.25, 0.26))
    assert result["statuses"]["HYBRID_STATUS"] != "SUPPORTED"
    assert result["INTERPRETATION_BRANCH"] == "B"
    assert analyze_rows(rows_for(0.35, 0.41, 0.25, 0.26))["INTERPRETATION_BRANCH"] == "C"


def test_runner_trains_every_width_and_seed_concurrently_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_completion_experiment.py")
    seen = []
    monkeypatch.setattr(module, "execute", lambda root, label, args, **kwargs: seen.append(args))
    config = {
        "seeds": [1, 2, 3],
        "variants": ["C0", "C1"],
        "primary_pair": ["C0", "C1"],
        "parallel_workers": 6,
        "infrastructure_retries": 2,
    }
    module.train_all(tmp_path, config, env=None, times={})
    assert len(seen) == 6 and {a[0] for a in seen} == {"scripts/train_rgbd_completion_carrier.py"}
    pairs = {(a[a.index("--variant") + 1], a[a.index("--seed") + 1]) for a in seen}
    assert pairs == {(v, s) for v in ("C0", "C1") for s in ("1", "2", "3")}
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)
