"""V10 RGB-only bounds prior: only the prior (and C2's grid) differs; 16^3 reproduces V5 C0."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_geometry_carrier_evaluation import fixture as evaluation_fixture
from test_mechanism_training import observation

from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import geometry_carrier
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.geometry_carrier_experiment import SceneData
from mcss.mechanism_pilot.optimization_bounds_contracts import (
    FrozenTrainingPrior,
    context_camera_bundle,
    frozen_gt_free_bounds,
)
from mcss.mechanism_pilot.rgb_prior_bounds_carrier import (
    GRIDS,
    PRIOR_FILES,
    SCENE_DATA,
    VARIANT_SPECS,
    Train72PriorSceneData,
    build_state,
    load_checkpoint,
    make_carrier,
    parameter_hash,
    save_checkpoint,
    scene_data,
    training_loss,
)
from mcss.mechanism_pilot.rgb_prior_bounds_evaluation import evaluate_model
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.types import Cameras


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context():
    torch.set_num_threads(1)
    observations = tuple(observation(i) for i in (0, 1, 2))
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    query = observation(5).camera
    cameras = Cameras(query.intrinsics[None, None], query.c2w[None, None], (8, 8))
    cameras = transform_cameras(cameras, torch.linalg.inv(observations[0].camera.c2w))
    return observations, bounds, cameras


def targets():
    return torch.full((1, 1, 3, 8, 8), 0.6), torch.full((1, 1, 1, 8, 8), 2.0), torch.arange(64)


def test_only_the_prior_and_the_secondary_grid_differ():
    assert {v: (s["prior"], s["grid"]) for v, s in VARIANT_SPECS.items()} == {
        "C0": ("FROZEN_V2_PRIOR", 32),
        "C1": ("TRAIN72_PRIOR", 32),
        "C2": ("TRAIN72_PRIOR", 16),
    }
    assert {s["model"] for s in VARIANT_SPECS.values()} == {"V5_C0_RGB"}
    assert {s["train"] for s in VARIANT_SPECS.values()} == {"TRAIN72"}
    assert set(GRIDS) == {16, 32}
    assert PRIOR_FILES["FROZEN_V2_PRIOR"] == "train_depth_prior.json"
    assert scene_data("C0") is SceneData
    assert scene_data("C1") is scene_data("C2") is Train72PriorSceneData
    carriers = {v: make_carrier(v, 3, "cpu") for v in VARIANT_SPECS}
    assert len({parameter_hash(c) for c in carriers.values()}) == 1
    # The V5 C0 draw, which is also the shared-parameter draw of the V5-V9 RGB-D carriers.
    assert parameter_hash(carriers["C0"]) == parameter_hash(v5.make_carrier("C0", 3, "cpu"))
    assert parameter_hash(carriers["C0"]) == parameter_hash(v5.make_carrier("C1", 3, "cpu"))
    for v, carrier in carriers.items():
        assert type(carrier) is DynamicSceneCarrier
        assert sum(p.numel() for p in carrier.parameters()) == 5125
        assert not any(name.startswith("depth_") for name, _ in carrier.named_parameters())
        assert carrier._candidate_points.shape[0] == VARIANT_SPECS[v]["grid"] ** 3


def test_grid16_reproduces_the_v5_c0_state_and_loss_exactly():
    observations, bounds, cameras = context()
    old = v5.make_carrier("C0", 11, "cpu")
    new = make_carrier("C2", 11, "cpu")
    assert hash_value(old.state_dict()) == hash_value(new.state_dict())
    rgbd = v5.RGBDContext(list(observations), torch.full((3, 8, 8), 2.0))
    first, _ = v5.build_state(old, rgbd, bounds, "same")
    second, _ = build_state(new, observations, bounds, "same")
    assert hash_scene_state(first) == hash_scene_state(second)
    rgb, depth, indices = targets()
    a, a_terms = geometry_carrier.training_loss(first, cameras, rgb, depth, indices, "C1")
    b, b_terms = training_loss(second, cameras, rgb, depth, indices, "C2")
    assert torch.equal(a, b)
    assert all(
        torch.equal(torch.as_tensor(a_terms[k]), torch.as_tensor(b_terms[k])) for k in a_terms
    )


def test_depth_never_enters_construction():
    observations, bounds, cameras = context()
    carrier = make_carrier("C1", 5, "cpu")
    rgbd = v5.RGBDContext(list(observations), torch.full((3, 8, 8), 2.0))
    with pytest.raises(TypeError):
        build_state(carrier, rgbd, bounds, "rgbd")
    with pytest.raises(TypeError):
        build_state(v5.make_carrier("C1", 5, "cpu"), observations, bounds, "bypass")
    coarse = DynamicSceneCarrier(CarrierConfig(grid_size=(8, 8, 8), token_count=512))
    with pytest.raises(ValueError):
        build_state(coarse, observations, bounds, "coarse")
    state, _ = build_state(carrier, observations, bounds, "ok")
    assert tuple(state.spatial_shape) == (32, 32, 32)
    with pytest.raises(ValueError):
        training_loss(state, cameras, *targets(), "C9")


@pytest.mark.parametrize("variant", ["C1", "C2"])
def test_states_train_and_checkpoints_roundtrip(tmp_path, variant):
    observations, bounds, cameras = context()
    carrier = make_carrier(variant, 5, "cpu")
    state, _ = build_state(carrier, observations, bounds, "fine")
    g = VARIANT_SPECS[variant]["grid"]
    assert tuple(state.spatial_shape) == (g, g, g)
    loss, _ = training_loss(state, cameras, *targets(), variant)
    loss.backward()
    grads = [p.grad for p in carrier.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    assert sum(float(x.abs().sum()) for x in grads) > 0
    path = tmp_path / "carrier.pt"
    save_checkpoint(path, carrier, variant=variant, seed=5, step=1, lock_sha256="x")
    restored, payload = load_checkpoint(path, "cpu")
    assert type(restored) is DynamicSceneCarrier
    assert payload["depth_bypass"] is False and payload["test_time_input"] == "RGB+CAMERA"
    assert hash_value(restored.state_dict()) == hash_value(carrier.state_dict())
    with pytest.raises(PermissionError):
        v5.load_checkpoint(path, "cpu")


def write_train72_prior(root):
    write_json(root / "train72_depth_prior.json", {"near_m": 0.3, "far_m": 12.0})
    return FrozenTrainingPrior(0.3, 12.0, sha(root / "train72_depth_prior.json"))


def test_train72_prior_scene_data_sets_camera_only_bounds(tmp_path):
    manifest, _ = evaluation_fixture(tmp_path)
    prior = write_train72_prior(tmp_path)
    record = next(r for r in manifest["scenes"] if r["split"] == "DEV")
    access = []
    frozen = SCENE_DATA["FROZEN_V2_PRIOR"](record, manifest, tmp_path, "cpu", access)
    wide = SCENE_DATA["TRAIN72_PRIOR"](record, manifest, tmp_path, "cpu", access)
    for role in ("A", "B"):
        bundle = context_camera_bundle(record, role, manifest["image_size"])
        expected = frozen_gt_free_bounds(bundle, prior).to(torch.float32)
        assert torch.equal(wide.bounds[role], expected)
        assert not torch.equal(wide.bounds[role], frozen.bounds[role])
        context = wide.context(role)
        assert not isinstance(context, v5.RGBDContext)
        assert all(type(o).__name__ == "OnlineObservation" for o in context)
    assert access and not any("depth" in e.get("channels", []) for e in access)


def test_evaluation_reads_no_depth_before_sealing(tmp_path):
    torch.set_num_threads(1)
    manifest, _ = evaluation_fixture(tmp_path)
    write_train72_prior(tmp_path)
    boxes = {}
    for variant in ("C0", "C1"):
        carrier = make_carrier(variant, 5, "cpu")
        out = tmp_path / f"evaluation_{variant}"
        evaluate_model(carrier, manifest, tmp_path, out, variant, 5, 100, "cpu", diagnostics=True)
        access = json.loads((out / "GT_access.json").read_text())
        marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_DEV_STATES_SEALED")
        assert not any("depth" in e.get("channels", []) for e in access[:marker])
        assert any("depth" in e.get("channels", []) for e in access[marker:])
        integrity = json.loads((out / "integrity.json").read_text())
        assert integrity["test_time_depth_used"] is False
        rows = json.loads((out / "query_results.json").read_text())
        assert {r["method"] for r in rows} >= {"direct", "spatial_shuffle", "wrong_scene", "zero"}
        boxes[variant] = json.loads((out / "state_hashes.json").read_text())
    key = next(iter(boxes["C0"]))
    assert boxes["C0"][key]["bounds"] != boxes["C1"][key]["bounds"]


def test_train72_prior_pools_every_train_frame_and_nothing_else(tmp_path):
    module = script("prepare_rgb_prior_bounds_experiment.py")
    scenes, values = [], []
    for i in range(72):
        path = tmp_path / f"d{i}.npy"
        depth = np.array([[np.nan, 0.0, 1.0 + i, 100.0]], dtype=np.float32)
        np.save(path, depth)
        values.append(depth[0, 2:])
        frames = [{"frame_id": 0, "depth": str(path)}]
        scenes.append({"scene_id": f"s{i:02d}", "split": "TRAIN", "frames": frames})
    np.save(tmp_path / "dev.npy", np.array([[1e6]], dtype=np.float32))
    frames = [{"frame_id": 0, "depth": str(tmp_path / "dev.npy")}]
    scenes.append({"scene_id": "dev", "split": "DEV", "frames": frames})
    source = tmp_path / "split.json"
    source.write_text("{}")
    prior = module.train72_prior({"scenes": scenes}, source)
    near, far = np.quantile(np.concatenate(values), [0.01, 0.99])
    assert (prior["near_m"], prior["far_m"]) == (float(near), float(far))
    assert prior["sample_count"] == 144 and prior["quantiles"] == [0.01, 0.99]
    assert {a["scene_id"] for a in prior["access"]} == {f"s{i:02d}" for i in range(72)}
    with pytest.raises(PermissionError):
        module.train72_prior({"scenes": scenes[1:]}, source)


def test_v10_runner_trains_only_v10_scripts_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgb_prior_bounds_experiment.py")
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
    assert [a[0] for a in seen] == ["scripts/train_rgb_prior_bounds_carrier.py"] * 3
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)


def test_v10_report_uses_the_amended_estimator_and_all_scene_reference(tmp_path):
    from test_geometry_carrier_statistics import fixture, region_fixture

    data = fixture()
    for row in data[0]:
        if row["variant"] == "C0" and row["method"] == "direct":
            # The frozen prior truncates; only the amended estimator accepts unequal hits.
            row["ray_hitfraction"] = 0.5
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
    script("analyze_rgb_prior_bounds_carrier.py").run(tmp_path)
    hits = json.loads((tmp_path / "ray_hit_fractions.json").read_text())["per_variant_scene"]
    assert set(hits["C0"].values()) == {0.5} and set(hits["C1"].values()) != {0.5}
    report = script("report_rgb_prior_bounds_carrier.py")
    fields = report.run(tmp_path)
    assert fields["PRIOR_BOUNDS_GAIN"] == "0.010000"
    assert fields["PRIOR_BOUNDS_STATUS"] == "SUPPORTED"
    assert fields["INTERPRETATION_BRANCH"] == "B" and fields["TEST_TIME_INPUT"] == "RGB+CAMERA"
    assert fields["MEAN_RAY_HIT_FRACTION"]["C0"] == "0.500000"
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
        "carriers": {"V10_C0": carrier(below), "V10_C1": carrier(below)},
        "sensitivity_excluding_no_hit_scenes": {
            "excluded_no_hit_scenes": ["x"],
            "carriers": {"V10_C0": carrier(above), "V10_C1": carrier(above)},
        },
    }
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    assert report.run(tmp_path)["INTERPRETATION_BRANCH"] == "B"
    reference["carriers"]["V10_C1"] = carrier(above)
    (tmp_path / "reference_results.json").write_text(json.dumps(reference))
    fields = report.run(tmp_path)
    assert fields["INTERPRETATION_BRANCH"] == "A" and fields["C1_ABOVE_REFERENCE_ALL_SCENES"]
