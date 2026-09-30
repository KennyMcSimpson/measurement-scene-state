"""Core B V1: stream rule, frozen-carrier identity, write mechanics and the sealed DEV boundary."""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from test_geometry_carrier_evaluation import fixture as evaluation_fixture
from test_mechanism_training import observation

from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot.rgbd_stream_evaluation import evaluate_streams
from mcss.mechanism_pilot.rgbd_stream_write import (
    CONTROLS,
    POLICIES,
    StreamRGBDCarrier,
    StreamRGBDLoader,
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
    model = StreamRGBDCarrier(v5.carrier_config())
    model.load_state_dict(v5.make_carrier("C1", seed, "cpu").state_dict())
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


def stream_fixture(root):
    manifest, _ = evaluation_fixture(root)
    for record in manifest["scenes"]:
        base = next(f for f in record["frames"] if f["frame_id"] == 4)
        for fid in (5, 6, 7, 10):
            frame = copy.deepcopy(base)
            frame["frame_id"] = fid
            for key, suffix in (("rgb", ".png"), ("depth", ".npy")):
                source = Path(base[key])
                if source.exists():
                    target = source.with_name(f"{record['scene_id']}_{fid}{suffix}")
                    shutil.copyfile(source, target)
                    frame[key] = str(target)
            pose = np.array(frame["c2w"])
            pose[0, 3] += 0.01 * fid
            frame["c2w"] = pose.tolist()
            record["frames"].append(frame)
        record["frames"].sort(key=lambda f: f["frame_id"])
    return manifest


def test_stream_rule_uses_frame_ids_only_and_never_shortens():
    record = {
        "roles": {"context_a": [0, 1, 2], "context_b": [0, 3, 4], "primary_query": [8, 9]},
        "frames": [{"frame_id": i} for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)],
    }
    assert stream_frame_ids(record, "A") == (5, 6, 7, 10)
    assert stream_frame_ids(record, "B") == (5, 6, 7, 10)
    record["frames"] = record["frames"][:-1]
    assert stream_frame_ids(record, "A") is None
    # free frames may precede warmup frames; the stream spreads evenly over all of them
    record = {
        "roles": {"context_a": [0, 6, 7], "context_b": [0, 12, 13], "primary_query": [14, 15]},
        "frames": [{"frame_id": i} for i in range(16)],
    }
    assert stream_frame_ids(record, "A") == (1, 4, 8, 11)


def test_frozen_carrier_and_no_stream_reproduce_the_static_v5_state():
    warm, stream, bounds, _ = contexts()
    model = carrier()
    reference = v5.make_carrier("C1", 5, "cpu")
    reference.load_state_dict(model.state_dict())
    expected, _ = v5.build_state(reference, warm, bounds, "static")
    rule = make_write_rule(model, 1)
    state, _, fast, _ = unroll(model, rule, warm, stream, bounds, "NO_STREAM", "static")
    assert hash_scene_state(state) == hash_scene_state(expected)
    assert fast.step == 0 and float(fast.delta_fuse.abs().sum()) == 0.0
    # three views never exceed the trained count, so the clamp is inert without a stream
    model.count_clamp = 3
    clamped, _ = v5.build_state(model, warm, bounds, "static")
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
    assert hashes["OFF_CLAMP3"] != hashes["OFF"]  # seven views exceed the trained count


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
    grads = [p.grad for p in rule.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in model.parameters())
    assert hash_value(model.state_dict()) == before
    path = tmp_path / "rule.pt"
    save_write_rule(path, rule, seed=3, step=1, lock_sha256="x", carrier_sha256="y")
    restored, payload = load_write_rule(path, "cpu")
    assert hash_value(restored.state_dict()) == hash_value(rule.state_dict())
    assert payload["slow_carrier_updated"] is False
    with pytest.raises(FileExistsError):
        save_write_rule(path, rule, seed=3, step=1, lock_sha256="x", carrier_sha256="y")


def test_stream_loader_reads_stream_frames_only(tmp_path):
    manifest = stream_fixture(tmp_path)
    record = next(r for r in manifest["scenes"] if r["split"] == "DEV")
    access = []
    loader = StreamRGBDLoader(
        record,
        manifest["image_size"],
        tmp_path,
        "cpu",
        access,
        allowed_scene_ids=[record["scene_id"]],
        holdout_scene_ids=["train_unused"],
    )
    stream = loader.stream("A")
    assert tuple(o.frame_id for o in stream) == (5, 6, 7, 10)
    assert {e["frame_id"] for e in access if "depth" in e.get("channels", [])} == {5, 6, 7, 10}
    assert {e["purpose"] for e in access} == {"STREAM_RGBD"}
    with pytest.raises(PermissionError):
        StreamRGBDLoader(
            record,
            manifest["image_size"],
            tmp_path,
            "cpu",
            [],
            allowed_scene_ids=["other"],
            holdout_scene_ids=[],
        )


def test_dev_evaluation_seals_every_policy_before_query_depth(tmp_path):
    manifest = stream_fixture(tmp_path)
    model = carrier()
    rules = {"trained": make_write_rule(model, 4), "untrained": make_write_rule(model, 4)}
    out = tmp_path / "evaluation"
    rows = evaluate_streams(model, rules, manifest, tmp_path, out, 4, "cpu")
    names = (*POLICIES, *CONTROLS)
    assert len(rows) == 2 * 2 * 2 * len(names)
    access = json.loads((out / "GT_access.json").read_text())
    marker = next(i for i, e in enumerate(access) if e.get("event") == "ALL_STREAM_STATES_SEALED")
    early = [e for e in access[:marker] if "depth" in e.get("channels", [])]
    assert {e["purpose"] for e in early} == {"CONTEXT_ONLY_RGBD", "STREAM_RGBD"}
    assert all(e["frame_id"] not in (8, 9) for e in early)
    assert all(e.get("scene_id") != "train_unused" for e in access)
    metadata = json.loads((out / "state_hashes.json").read_text())
    donors = {m["donor_scene_id"] for k, m in metadata.items() if k.endswith("ALL_WRONG_SCENE")}
    assert donors == {"capacity0", "capacity1"}


def test_analysis_gates_and_branches():
    analyze = script("analyze_rgbd_stream_write.py").analyze
    scenes = [f"s{i}" for i in range(8)]
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
    rows = []
    for seed in (1, 2):
        for i, sid in enumerate(scenes):
            for role in ("A", "B"):
                for qid in (8, 9):
                    for policy, value in base.items():
                        noise = 0.002 * ((i + qid + seed) % 3)
                        rows.append(
                            {
                                "seed": seed,
                                "scene_id": sid,
                                "role": role,
                                "query_id": qid,
                                "policy": policy,
                                "depth_absrel": value + noise,
                                "depth_delta1": 0.5,
                            }
                        )
    result = analyze(rows, [1, 2], True)
    assert result["INTERPRETATION_BRANCH"] == "A"
    assert result["contrasts"]["WRITE_GAIN"]["mean"] == pytest.approx(0.09)
    assert result["statuses"]["WRITE_SPECIFICITY_STATUS"] == "SUPPORTED"
    for row in rows:
        if row["policy"] == "ALL":
            row["depth_absrel"] += 0.08  # still better than OFF, worse than NO_STREAM
    result = analyze(rows, [1, 2], True)
    assert result["statuses"]["WRITE_STATUS"] == "SUPPORTED"
    assert result["statuses"]["STREAM_STATUS"] == "HARMFUL"
    assert result["INTERPRETATION_BRANCH"] == "B"
    assert analyze(rows, [1, 2], False)["INTERPRETATION_BRANCH"] == "C"


def test_runner_trains_only_stream_write_scripts_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_stream_write.py")
    seen = []
    monkeypatch.setattr(module, "execute", lambda root, label, args, env, times: seen.append(args))
    module.train_all(tmp_path, {"seeds": [1, 2], "parallel_workers": 2}, None, {})
    assert sorted(a[a.index("--seed") + 1] for a in seen) == ["1", "2"]
    assert all(a[0] == "scripts/train_rgbd_stream_write.py" for a in seen)
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)
