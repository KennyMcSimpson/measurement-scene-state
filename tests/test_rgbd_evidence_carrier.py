"""V5 RGB-D evidence carrier: geometry of the depth cue, matched design and data boundary."""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from test_geometry_carrier_evaluation import fixture as evaluation_fixture
from test_mechanism_training import observation

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.types import OnlineObservation, hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.rgbd_evidence_carrier import (
    VARIANT_SPECS,
    RGBDContext,
    RGBDEvidenceCarrier,
    build_state,
    depth_statistics,
    load_checkpoint,
    make_carrier,
    parameter_hash,
    save_checkpoint,
    training_loss,
)
from mcss.mechanism_pilot.rgbd_evidence_evaluation import evaluate_model
from mcss.training.grounded import build_grounded_state
from mcss.types import Cameras

SIGMA = torch.tensor(math.log(0.5))
EDGE = torch.tensor(0.5)


def camera(height=32, width=40, focal=30.0):
    cx, cy = (width - 1) / 2, (height - 1) / 2
    intrinsics = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    return Cameras(intrinsics, torch.eye(4), (height, width))


def ray_distance(cam, z):
    """Measured ray distance of a z-depth map [H, W] (the dataset convention)."""
    height, width = cam.image_size
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    directions = torch.stack((u, v, torch.ones_like(u)), -1) @ torch.linalg.inv(cam.intrinsics).T
    return z * directions.norm(dim=-1)


def view(cam):
    return OnlineObservation("s", 0, torch.full((3, *cam.image_size), 0.5), cam)


def test_depth_cue_localizes_a_measured_surface_and_its_free_space():
    cam = camera()
    depth = ray_distance(cam, torch.full(cam.image_size, 2.0))
    points = torch.tensor([[0.0, 0.0, z] for z in (1.0, 1.5, 2.0, 2.5, 3.0)])
    stats = depth_statistics([view(cam)], depth[None], points, SIGMA, EDGE)
    surface, free, valid = stats.unbind(-1)
    assert torch.isfinite(stats).all() and torch.all(valid == 1.0)
    assert surface[2] > 0.99 and surface[0] < 1e-3 and surface[4] < 1e-3
    assert surface[1] == pytest.approx(math.exp(-2.0), rel=0.01)
    assert free[0] > 0.98 and abs(float(free[2]) - 0.5) < 0.01 and free[4] < 0.02


@pytest.mark.parametrize("axis", [0, 1])
def test_depth_lookup_follows_image_u_and_v(axis):
    """Near half-plane (coordinate `axis` < 0) at z=1.5 in front of a far half at z=3."""
    cam = camera()
    height, width = cam.image_size
    v, u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    near_side = (u if axis == 0 else v) < (width if axis == 0 else height) / 2
    depth = ray_distance(cam, torch.where(near_side, 1.5, 3.0))

    def point(sign, z):
        p = [0.0, 0.0, z]
        p[axis] = sign * 0.4
        return p

    # (near side, z=1.5) (far side, z=3) (far side, z=1.5) (near side, z=3)
    points = torch.tensor([point(-1, 1.5), point(1, 3.0), point(1, 1.5), point(-1, 3.0)])
    surface, free, _ = depth_statistics([view(cam)], depth[None], points, SIGMA, EDGE).unbind(-1)
    assert surface[0] > 0.9 and surface[1] > 0.9
    assert surface[2] < 0.1 and surface[3] < 0.1
    assert free[2] > 0.9 and free[3] < 0.1


def test_unusable_views_contribute_nothing_and_stay_finite():
    cam = camera()
    log_sigma = SIGMA.clone().requires_grad_(True)
    depth = torch.zeros(1, *cam.image_size)
    points = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [50.0, 0.0, 1.0]])
    stats = depth_statistics([view(cam)], depth, points, log_sigma, EDGE)
    assert torch.equal(stats, torch.zeros_like(stats))
    measured = ray_distance(cam, torch.full(cam.image_size, 2.0))[None]
    stats = depth_statistics([view(cam)], measured, points, log_sigma, EDGE)
    assert torch.equal(stats[1:], torch.zeros_like(stats[1:]))
    stats[:, :2].sum().backward()
    assert torch.isfinite(log_sigma.grad)


def context():
    torch.set_num_threads(1)
    observations = [observation(i) for i in (0, 1, 2)]
    depths = torch.full((3, 8, 8), 2.0)
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    query = observation(5).camera
    cameras = Cameras(query.intrinsics[None, None], query.c2w[None, None], (8, 8))
    cameras = transform_cameras(cameras, torch.linalg.inv(observations[0].camera.c2w))
    return RGBDContext(observations, depths), bounds, cameras


def test_depth_carrier_starts_exactly_equal_to_the_dense_baseline():
    rgbd, bounds, _ = context()
    base = make_carrier("C0", 11, "cpu")
    depth = make_carrier("C1", 11, "cpu")
    assert type(base) is DynamicSceneCarrier and isinstance(depth, RGBDEvidenceCarrier)
    assert parameter_hash(base) == parameter_hash(depth)
    assert parameter_hash(make_carrier("C2", 11, "cpu")) == parameter_hash(base)
    assert sum(p.numel() for p in base.parameters()) == 5125
    assert sum(p.numel() for p in depth.parameters()) == 5150
    first, _ = build_state(base, rgbd, bounds, "same")
    second, _ = build_state(depth, rgbd, bounds, "same")
    assert hash_scene_state(first) == hash_scene_state(second)


def test_loss_gradient_reaches_the_depth_bypass_and_sigma():
    rgbd, bounds, cameras = context()
    carrier = make_carrier("C1", 5, "cpu")

    def loss():
        state, _ = build_state(carrier, rgbd, bounds, "grad")
        value, _ = training_loss(
            state,
            cameras,
            torch.full((1, 1, 3, 8, 8), 0.6),
            torch.full((1, 1, 1, 8, 8), 2.0),
            torch.arange(64),
            "C1",
        )
        return value

    loss().backward()
    grad = carrier.depth_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    carrier.zero_grad()
    with torch.no_grad():
        carrier.depth_weight.fill_(0.1)
    loss().backward()
    assert torch.isfinite(carrier.depth_log_sigma.grad) and carrier.depth_log_sigma.grad != 0


def test_context_contract_and_state_entry_are_enforced():
    rgbd, bounds, _ = context()
    carrier = make_carrier("C1", 5, "cpu")
    with pytest.raises(ValueError):
        RGBDContext(list(rgbd), torch.zeros(3, 4, 4))
    with pytest.raises(ValueError):
        RGBDContext(list(rgbd), torch.zeros(2, 8, 8))
    with pytest.raises(TypeError):
        build_state(carrier, tuple(rgbd), bounds, "plain")
    with pytest.raises(PermissionError):
        build_grounded_state(carrier, tuple(rgbd), "bypass")
    assert list(rgbd)[1].frame_id == 1 and rgbd.frame_ids == (0, 1, 2)


def test_variant_specs_and_checkpoint_roundtrip(tmp_path):
    assert {v: s["loss"] for v, s in VARIANT_SPECS.items()} == {"C0": "C1", "C1": "C1", "C2": "C0"}
    assert {v: s["depth"] for v, s in VARIANT_SPECS.items()} == {
        "C0": False,
        "C1": True,
        "C2": True,
    }
    for variant in ("C0", "C1"):
        carrier = make_carrier(variant, 3, "cpu")
        with torch.no_grad():
            for p in carrier.parameters():
                p.add_(0.01)
        path = tmp_path / f"{variant}.pt"
        save_checkpoint(path, carrier, variant=variant, seed=3, step=1, lock_sha256="x")
        restored, payload = load_checkpoint(path, "cpu")
        assert type(restored) is type(carrier)
        assert hash_value(restored.state_dict()) == hash_value(carrier.state_dict())
        assert payload["variant"] == variant
        assert payload["test_time_input"] == "RGB+DEPTH+CAMERA"
    payload = torch.load(tmp_path / "C1.pt", weights_only=True)
    payload["depth_bypass"] = False
    torch.save(payload, tmp_path / "forged.pt")
    with pytest.raises(PermissionError):
        load_checkpoint(tmp_path / "forged.pt", "cpu")


def test_dev_states_read_context_depth_only_and_query_depth_after_seal(tmp_path):
    torch.set_num_threads(1)
    manifest, _ = evaluation_fixture(tmp_path)
    carrier = make_carrier("C1", 5, "cpu")
    with torch.no_grad():
        carrier.depth_weight.fill_(0.05)
    out = tmp_path / "evaluation"
    result = evaluate_model(carrier, manifest, tmp_path, out, "C1", 5, 100, "cpu", diagnostics=True)
    assert len(result["query_rows"]) == 40 and len(result["context_rows"]) == 12
    access = json.loads((out / "GT_access.json").read_text())
    position = next(i for i, e in enumerate(access) if e.get("event") == "ALL_DEV_STATES_SEALED")
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    before = [e for e in access[:position] if "depth" in e.get("channels", [])]
    assert before, "context depth must reach state construction"
    for event in before:
        roles = records[event["scene_id"]]["roles"]
        assert event["purpose"] == "CONTEXT_ONLY_RGBD"
        assert event["frame_id"] in set(roles["context_a"]) | set(roles["context_b"])
        assert event["frame_id"] not in roles["primary_query"]
    assert all(e.get("scene_id", "") != "train_unused" for e in access)
    integrity = json.loads((out / "integrity.json").read_text())
    assert integrity["test_time_depth_used"] is True


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_runner_trains_only_v5_scripts(tmp_path, monkeypatch):
    module = script("run_rgbd_evidence_experiment.py")
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
    assert seen == ["scripts/train_rgbd_evidence_carrier.py"] * 3


def test_v5_report_uses_depth_semantics(tmp_path):
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
    script("analyze_rgbd_evidence_carrier.py").run(tmp_path)
    report = script("report_rgbd_evidence_carrier.py")
    fields = report.run(tmp_path)
    assert fields["DEPTH_GAIN"] == "0.010000"
    assert fields["DEPTH_STATUS"] == "SUPPORTED"
    assert fields["SCENE_SPECIFICITY_STATUS"] == "SUPPORTED"
    # Without the geometry-free reference, branch A is unreachable.
    assert fields["INTERPRETATION_BRANCH"] == "B"
    assert fields["C1_ABOVE_GEOMETRY_FREE_REFERENCE"] is False
    assert fields["TEST_TIME_DEPTH_USED"] is True
    assert fields["CARRIER_REFERENCE_STATUS"] == "NOT_RECORDED"
    assert not fields["DYNAMIC_TTT_NEXT_STAGE_ALLOWED"]
    text = (tmp_path / "README.md").read_text()
    assert "DEPTH_GAIN" in text and all(f"**{i}. " in text for i in range(1, 13))
    stats = {"mean": 0.1, "ci95": [0.05, 0.15]}
    cell = {
        "reference_label": "ABOVE",
        "vs_REF_TRAIN_ABSREL_OPTIMAL": stats,
        "vs_REF_TRAIN_MEDIAN": stats,
    }
    carrier = {
        r: {m: cell for m in ("depth_absrel", "depth_delta1")}
        for r in ("RAW", "OPACITY_NORMALIZED", "MEDIAN_SCALED")
    }
    reference = {
        "summary": {"CARRIER_REFERENCE_STATUS": "SOME_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE"},
        "carriers": {"V5_C0": carrier, "V5_C1": carrier},
        "reference_values": {"REF_TRAIN_ABSREL_OPTIMAL": 2.0, "REF_TRAIN_MEDIAN": 4.0},
        "primary_reference": {
            "depth_absrel": "REF_TRAIN_ABSREL_OPTIMAL",
            "depth_delta1": "REF_TRAIN_MEDIAN",
        },
    }
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    fields = report.run(tmp_path)
    assert fields["INTERPRETATION_BRANCH"] == "A"
    assert fields["C1_ABOVE_GEOMETRY_FREE_REFERENCE"] is True
