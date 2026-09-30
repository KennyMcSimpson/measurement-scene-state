"""Core B V2: the V1 stream mechanics on the frozen 32^3 V9 carrier, and the V2 primary gates."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_mechanism_training import observation
from test_rgbd_stream_write import stream_fixture

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot import rgbd_stream_write as v1
from mcss.mechanism_pilot.rgbd_stream_evaluation_v2 import evaluate_streams
from mcss.mechanism_pilot.rgbd_stream_write_v2 import (
    CONTROLS,
    POLICIES,
    StreamResolutionCarrier,
    load_frozen_carrier,
    load_write_rule,
    make_write_rule,
    save_write_rule,
    stream_frame_ids,
    training_loss,
    unroll,
)
from mcss.types import Cameras


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def carrier(seed=5):
    model = StreamResolutionCarrier(v8.carrier_config("C1"))
    model.load_state_dict(v8.make_carrier("C1", seed, "cpu").state_dict())
    with torch.no_grad():
        model.depth_weight.fill_(0.05)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def contexts(n_warm=3, n_stream=4):
    torch.set_num_threads(1)
    observations = [observation(i) for i in range(n_warm + n_stream)]
    depths = torch.full((n_warm + n_stream, 8, 8), 2.0)
    warm = v5.RGBDContext(observations[:n_warm], depths[:n_warm])
    stream = v5.RGBDContext(observations[n_warm:], depths[n_warm:])
    bounds = torch.tensor([[-2.0, -2.0, 0.1], [2.0, 2.0, 4.0]])
    query = observation(9).camera
    cameras = Cameras(query.intrinsics[None, None], query.c2w[None, None], (8, 8))
    cameras = transform_cameras(cameras, torch.linalg.inv(observations[0].camera.c2w))
    return warm, stream, bounds, cameras


def test_the_stream_rule_and_policies_are_core_b_v1s():
    assert stream_frame_ids is v1.stream_frame_ids
    assert POLICIES == v1.POLICIES and CONTROLS == v1.CONTROLS
    record = {
        "roles": {"context_a": [0, 1, 2], "context_b": [0, 3, 4], "primary_query": [8, 9]},
        "frames": [{"frame_id": i} for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)],
    }
    assert stream_frame_ids(record, "A") == (5, 6, 7, 10)


def test_no_stream_reproduces_the_static_v9_state():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    reference = v8.make_carrier("C1", 5, "cpu")
    reference.load_state_dict(model.state_dict())
    expected, _ = v8.build_state(reference, warm, bounds, "static")
    rule = make_write_rule(model, 1)
    state, _, fast, _ = unroll(model, rule, warm, stream, bounds, "NO_STREAM", "static")
    assert tuple(state.spatial_shape) == (32, 32, 32)
    assert hash_scene_state(state) == hash_scene_state(expected)
    assert fast.step == 0 and float(fast.delta_fuse.abs().sum()) == 0.0
    model.count_clamp = 3  # three views never exceed the clamp
    clamped, _ = v8.build_state(model, warm, bounds, "static")
    model.count_clamp = None
    assert hash_scene_state(clamped) == hash_scene_state(expected)


def test_policies_write_only_the_matrices_they_name():
    warm, stream, bounds, _ = contexts()
    model, rule = carrier(), make_write_rule(carrier(), 2)
    results = {p: unroll(model, rule, warm, stream, bounds, p, "e") for p in POLICIES}
    off_fast = results["OFF"][2]
    assert off_fast.step == 0 and float(off_fast.delta_complete.abs().sum()) == 0.0
    assert results["ALL"][2].step == len(stream)
    assert float(results["FUSE"][2].delta_complete.detach().abs().sum()) == 0.0
    assert float(results["FUSE"][2].delta_fuse.detach().abs().sum()) > 0.0
    assert float(results["COMPLETE"][2].delta_fuse.detach().abs().sum()) == 0.0
    hashes = {p: hash_scene_state(r[0]) for p, r in results.items()}
    assert hashes["OFF"] != hashes["NO_STREAM"] and hashes["ALL"] != hashes["OFF"]
    assert hashes["OFF_CLAMP3"] != hashes["OFF"]
    with pytest.raises(TypeError):
        unroll(v1.StreamRGBDCarrier(v5.carrier_config()), rule, warm, stream, bounds, "OFF", "x")


def test_write_rule_learns_through_the_frozen_carrier(tmp_path):
    warm, stream, bounds, cameras = contexts()
    model, rule = carrier(), make_write_rule(carrier(), 3)
    before = hash_value(model.state_dict())
    state, _, _, _ = unroll(model, rule, warm, stream, bounds, "ALL", "grad")
    loss, _ = training_loss(
        state,
        cameras,
        torch.full((1, 1, 3, 8, 8), 0.6),
        torch.full((1, 1, 1, 8, 8), 2.0),
        torch.arange(64),
    )
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in rule.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert hash_value(model.state_dict()) == before
    path = tmp_path / "rule.pt"
    save_write_rule(path, rule, seed=3, step=1, lock_sha256="x", carrier_sha256="y")
    restored, payload = load_write_rule(path, "cpu")
    assert hash_value(restored.state_dict()) == hash_value(rule.state_dict())
    assert payload["schema"] == "mcss.rgbd_stream_write_rule.v2"
    with pytest.raises(PermissionError):
        v1.load_write_rule(path, "cpu")


def test_only_32_grid_v8_checkpoints_load_as_frozen_carriers(tmp_path):
    for variant, ok in (("C1", True), ("C0", False)):
        path = tmp_path / f"{variant}.pt"
        v8.save_checkpoint(
            path,
            v8.make_carrier(variant, 7, "cpu"),
            variant=variant,
            seed=7,
            step=1,
            lock_sha256="x",
        )
        if ok:
            model, _ = load_frozen_carrier(path, "cpu")
            assert isinstance(model, StreamResolutionCarrier) and not model.training
            assert not any(p.requires_grad for p in model.parameters())
        else:
            with pytest.raises(PermissionError):
                load_frozen_carrier(path, "cpu")


def test_dev_evaluation_seals_every_policy_before_query_depth(tmp_path):
    manifest = stream_fixture(tmp_path)
    model = carrier()
    rules = {"trained": make_write_rule(model, 4), "untrained": make_write_rule(model, 4)}
    out = tmp_path / "evaluation"
    rows = evaluate_streams(model, rules, manifest, tmp_path, out, 4, "cpu")
    assert len(rows) == 2 * 2 * 2 * len((*POLICIES, *CONTROLS))
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)


def rows_for(values):
    rows = []
    for seed in (1, 2):
        for i in range(8):
            for role in ("A", "B"):
                for qid in (8, 9):
                    for policy, value in values.items():
                        rows.append(
                            {
                                "seed": seed,
                                "scene_id": f"s{i}",
                                "role": role,
                                "query_id": qid,
                                "policy": policy,
                                "depth_absrel": value + 0.002 * ((i + qid + seed) % 3),
                                "depth_delta1": 0.5,
                            }
                        )
    return rows


def test_v2_primary_gates_set_the_branch():
    analyze = script("analyze_rgbd_stream_write_v2.py").analyze
    base = {
        "NO_STREAM": 0.30,
        "OFF": 0.34,
        "OFF_CLAMP3": 0.31,
        "FUSE": 0.30,
        "COMPLETE": 0.32,
        "ALL": 0.25,
        "ALL_UNTRAINED": 0.34,
        "ALL_WRONG_SCENE": 0.40,
    }
    result = analyze(rows_for(base), [1, 2], True, "OFF")
    assert result["contrasts"]["BEYOND_BASELINE"]["mean"] == pytest.approx(0.09)
    assert result["statuses"]["BEYOND_BASELINE_STATUS"] == "SUPPORTED"
    assert result["statuses"]["SPECIFICITY_STATUS"] == "SUPPORTED"
    assert result["INTERPRETATION_BRANCH"] == "A" and result["baseline_policy"] == "OFF"
    # beats the baseline, but another scene's fast weights do just as well
    result = analyze(rows_for({**base, "ALL_WRONG_SCENE": 0.25}), [1, 2], True, "OFF")
    assert result["statuses"]["SPECIFICITY_STATUS"] != "SUPPORTED"
    assert result["INTERPRETATION_BRANCH"] == "B"
    # the clamp baseline beats the writes (the Core B V1 pattern)
    result = analyze(rows_for({**base, "ALL": 0.33}), [1, 2], True, "OFF_CLAMP3")
    assert result["contrasts"]["BEYOND_BASELINE"]["mean"] < 0
    assert result["INTERPRETATION_BRANCH"] == "C"
    with pytest.raises(ValueError):
        analyze(rows_for(base), [1, 2], True, "NO_STREAM")


def test_runner_trains_only_v2_scripts_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_stream_write_v2.py")
    seen = []
    monkeypatch.setattr(module, "execute", lambda root, label, args, env, times: seen.append(args))
    module.train_all(tmp_path, {"seeds": [1, 2], "parallel_workers": 2}, None, {})
    assert sorted(a[a.index("--seed") + 1] for a in seen) == ["1", "2"]
    assert all(a[0] == "scripts/train_rgbd_stream_write_v2.py" for a in seen)
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)
