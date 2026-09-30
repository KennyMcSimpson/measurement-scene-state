"""V4 plane-sweep carrier: geometric correctness of the cue and the matched single-factor design."""

import importlib.util
import json
import math
from pathlib import Path

import torch
from test_mechanism_training import observation

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.types import OnlineObservation, hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.plane_sweep_carrier import (
    VARIANT_SPECS,
    PlaneSweepCarrier,
    build_state,
    load_checkpoint,
    make_carrier,
    parameter_hash,
    plane_sweep_statistics,
    save_checkpoint,
    training_loss,
)
from mcss.types import Cameras

NEAR, FAR = 1.0, 4.0


def texture(x, y):
    return torch.stack(
        (
            0.5 + 0.4 * torch.sin(9 * x + 4 * y),
            0.5 + 0.4 * torch.cos(11 * x - 3 * y),
            0.5 + 0.4 * torch.sin(13 * y + 2 * x),
        )
    )


def plane_view(center_x, frame, height=64, width=80, plane_z=2.0, focal=80.0):
    """Analytic render of a textured fronto-parallel plane z=plane_z seen along +z."""
    cx, cy = (width - 1) / 2, (height - 1) / 2
    intrinsics = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    x = center_x + (u - cx) / focal * plane_z
    y = (v - cy) / focal * plane_z
    pose = torch.eye(4)
    pose[0, 3] = center_x
    return OnlineObservation(
        "plane", frame, texture(x, y), Cameras(intrinsics, pose, (height, width))
    )


def test_plane_sweep_localizes_a_known_surface_and_its_free_space():
    # 4x area downsample to 16x20; baselines give 2-4 px disparity steps between test depths.
    views = [plane_view(x, i) for i, x in enumerate((0.0, -0.4, 0.4))]
    depths = torch.tensor([1.2, 1.6, 2.0, 2.4, 3.2])
    points = torch.stack((torch.zeros(5), torch.zeros(5), depths), -1)
    stats = plane_sweep_statistics(
        views, points, NEAR, FAR, torch.tensor(math.log(0.01)), planes=16, size=(16, 20)
    )
    surface, free, valid = stats.unbind(-1)
    assert torch.isfinite(stats).all()
    assert torch.all(valid == 1.0)
    assert surface[2] > 2 * max(surface[0], surface[4])
    assert free[0] > 0.8 and free[4] < 0.2
    assert free[0] > free[2] > free[4]


def test_single_view_contributes_no_cue_and_temperature_gets_gradient():
    views = [plane_view(x, i) for i, x in enumerate((0.0, -0.4, 0.4))]
    points = torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 1.5]])
    log_tau = torch.tensor(math.log(0.01), requires_grad=True)
    single = plane_sweep_statistics(views[:1], points, NEAR, FAR, log_tau, planes=16, size=(16, 20))
    assert torch.equal(single, torch.zeros_like(single))
    stats = plane_sweep_statistics(views, points, NEAR, FAR, log_tau, planes=16, size=(16, 20))
    stats[:, 0].sum().backward()
    assert log_tau.grad is not None and torch.isfinite(log_tau.grad) and log_tau.grad != 0


def context():
    torch.set_num_threads(1)
    observations = [observation(i) for i in (0, 1, 2)]
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    query = observation(5).camera
    cameras = Cameras(query.intrinsics[None, None], query.c2w[None, None], (8, 8))
    cameras = transform_cameras(cameras, torch.linalg.inv(observations[0].camera.c2w))
    return observations, bounds, cameras


def test_sweep_carrier_starts_exactly_equal_to_the_dense_baseline():
    observations, bounds, _ = context()
    base = make_carrier("C0", 11, "cpu", 0.2, 6.0)
    sweep = make_carrier("C1", 11, "cpu", 0.2, 6.0)
    assert type(base) is DynamicSceneCarrier and isinstance(sweep, PlaneSweepCarrier)
    assert parameter_hash(base) == parameter_hash(sweep)
    assert parameter_hash(make_carrier("C2", 11, "cpu", 0.2, 6.0)) == parameter_hash(base)
    assert sum(p.numel() for p in base.parameters()) == 5125
    assert sum(p.numel() for p in sweep.parameters()) == 5174
    first, _ = build_state(base, observations, bounds, "same")
    second, _ = build_state(sweep, observations, bounds, "same")
    assert hash_scene_state(first) == hash_scene_state(second)


def test_surface_loss_gradient_reaches_the_sweep_bypass():
    observations, bounds, cameras = context()
    carrier = make_carrier("C1", 5, "cpu", 0.2, 6.0)
    state, _ = build_state(carrier, observations, bounds, "grad")
    loss, _ = training_loss(
        state,
        cameras,
        torch.full((1, 1, 3, 8, 8), 0.6),
        torch.full((1, 1, 1, 8, 8), 2.0),
        torch.arange(64),
        "C1",
    )
    loss.backward()
    grad = carrier.sweep_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_variant_losses_follow_the_frozen_mapping():
    assert {v: s["loss"] for v, s in VARIANT_SPECS.items()} == {"C0": "C1", "C1": "C1", "C2": "C0"}
    assert {v: s["sweep"] for v, s in VARIANT_SPECS.items()} == {
        "C0": False,
        "C1": True,
        "C2": True,
    }


def test_checkpoint_roundtrip_keeps_class_and_weights(tmp_path):
    for variant in ("C0", "C1"):
        carrier = make_carrier(variant, 3, "cpu", 0.2, 6.0)
        with torch.no_grad():
            for p in carrier.parameters():
                p.add_(0.01)
        path = tmp_path / f"{variant}.pt"
        save_checkpoint(path, carrier, variant=variant, seed=3, step=1, lock_sha256="x")
        restored, payload = load_checkpoint(path, "cpu")
        assert type(restored) is type(carrier)
        assert hash_value(restored.state_dict()) == hash_value(carrier.state_dict())
        assert payload["variant"] == variant


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_runner_trains_only_v4_scripts(tmp_path, monkeypatch):
    module = script("run_plane_sweep_experiment.py")
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
    assert seen == ["scripts/train_plane_sweep_carrier.py"] * 3


def test_v4_report_uses_sweep_semantics(tmp_path):
    from test_geometry_carrier_statistics import fixture, region_fixture

    data = fixture()
    values = {
        "scene_split": data[3],
        "training_contract": {"seeds": data[4]["seeds"], "steps": 2000},
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
    script("analyze_plane_sweep_carrier.py").run(tmp_path)
    fields = script("report_plane_sweep_carrier.py").run(tmp_path)
    assert fields["SWEEP_GAIN"] == "0.010000"
    assert fields["SWEEP_STATUS"] == "SUPPORTED"
    assert fields["SCENE_SPECIFICITY_STATUS"] == "SUPPORTED"
    assert fields["INTERPRETATION_BRANCH"] == "A"
    assert fields["FRESH_QUALIFICATION_INCLUDED"] is False
    assert not fields["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    text = (tmp_path / "README.md").read_text()
    assert "SWEEP_GAIN" in text and all(f"**{i}. " in text for i in range(1, 12))
