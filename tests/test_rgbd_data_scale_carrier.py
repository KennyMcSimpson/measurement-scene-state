"""V14 data scale: only the TRAIN set differs; both variants are the V11 C1 draw; gates, runner."""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch
from test_mechanism_training import observation

from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.types import hash_scene_state, hash_value
from mcss.geometry import transform_cameras
from mcss.mechanism_pilot import rgbd_completion_carrier as v11
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot.rgbd_data_scale_carrier import (
    VARIANT_SPECS,
    build_state,
    load_checkpoint,
    make_carrier,
    save_checkpoint,
    train_records,
    training_loss,
)
from mcss.types import Cameras

ROOT = Path(__file__).parents[1]


def script(name):
    path = ROOT / "scripts" / name
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


def test_both_variants_are_the_v11_c1_draw_and_differ_only_by_their_train_set():
    assert {v: (s["hidden"], s["expansion"]) for v, s in VARIANT_SPECS.items()} == {
        "C0": (32, 64),
        "C1": (32, 64),
    }
    assert {v: s["train"] for v, s in VARIANT_SPECS.items()} == {"C0": "TRAIN72", "C1": "TRAIN_EXT"}
    reference = hash_value(v11.make_carrier("C1", 3, "cpu").state_dict())
    for variant in VARIANT_SPECS:
        carrier = make_carrier(variant, 3, "cpu")
        assert hash_value(carrier.state_dict()) == reference
        assert sum(p.numel() for p in carrier.parameters()) == 64238


def test_train_records_follow_the_v14_table_and_not_the_v8_table():
    manifest = {"scenes": [{"scene_id": f"t{i:02d}", "split": "TRAIN"} for i in range(30)]}
    manifest["scenes"].append({"scene_id": "d00", "split": "DEV"})
    config = {
        "train_sets": {
            "TRAIN72": [f"t{i:02d}" for i in range(24)],
            "TRAIN_EXT": [f"t{i:02d}" for i in range(30)],
        }
    }
    c0, c1 = train_records(manifest, config, "C0"), train_records(manifest, config, "C1")
    assert [r["scene_id"] for r in c0] == config["train_sets"]["TRAIN72"]
    assert [r["scene_id"] for r in c1] == config["train_sets"]["TRAIN_EXT"]
    # The V8 helper resolves every variant to TRAIN72: reusing it would silently drop TRAIN-EXT.
    assert len(v8.train_records(manifest, config, "C1")) == 24
    config["train_sets"]["TRAIN_EXT"].append("missing")
    with pytest.raises(PermissionError):
        train_records(manifest, config, "C1")


def test_both_variants_reproduce_the_v11_c1_state_and_loss_exactly():
    rgbd, bounds, cameras = context()
    old = v11.make_carrier("C1", 11, "cpu")
    with torch.no_grad():
        old.depth_weight.fill_(0.2)
    first, _ = v11.build_state(old, rgbd, bounds, "same")
    a, _ = v11.training_loss(first, cameras, *targets(), "C1")
    for variant in VARIANT_SPECS:
        new = make_carrier(variant, 11, "cpu")
        with torch.no_grad():
            new.depth_weight.fill_(0.2)
        second, _ = build_state(new, rgbd, bounds, "same")
        assert hash_scene_state(first) == hash_scene_state(second)
        b, _ = training_loss(second, cameras, *targets(), variant)
        assert torch.equal(a, b)


def test_checkpoints_roundtrip_under_the_v14_schema_only(tmp_path):
    rgbd, bounds, cameras = context()
    carrier = make_carrier("C1", 5, "cpu")
    state, _ = build_state(carrier, rgbd, bounds, "ext")
    assert tuple(state.spatial_shape) == (32, 32, 32)
    loss, _ = training_loss(state, cameras, *targets(), "C1")
    loss.backward()
    assert carrier.refinement[0].weight.grad.abs().sum() > 0
    path = tmp_path / "ext.pt"
    save_checkpoint(path, carrier, variant="C1", seed=5, step=1, lock_sha256="x")
    restored, payload = load_checkpoint(path, "cpu")
    assert payload["config"]["hidden_dim"] == 32 and payload["variant"] == "C1"
    assert hash_value(restored.state_dict()) == hash_value(carrier.state_dict())
    with pytest.raises(PermissionError):
        v11.load_checkpoint(path, "cpu")
    old = tmp_path / "v11.pt"
    v11.save_checkpoint(
        old, v11.make_carrier("C1", 5, "cpu"), variant="C1", seed=5, step=1, lock_sha256="v11"
    )
    with pytest.raises(PermissionError):
        load_checkpoint(old, "cpu")
    odd = v8.ResolutionRGBDCarrier(
        CarrierConfig(grid_size=(32, 32, 32), token_count=32**3, hidden_dim=64, expansion_dim=128)
    )
    with pytest.raises(ValueError):
        build_state(odd, rgbd, bounds, "odd")


def test_dev_curve_evaluations_do_not_save_states():
    source = (ROOT / "src/mcss/mechanism_pilot/rgbd_data_scale_evaluation.py").read_text()
    saves = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "diagnostics"
        and any("states.pt" in ast.unparse(child) for child in node.body)
    ]
    assert len(saves) == 1
    assert source.count('out / "states.pt")') == 2  # the guarded save and the guarded hash
    assert 'sha(out / "states.pt") if diagnostics else None' in source


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


def rows_for(c0_far, c1_far, harmonic_all, hfill_all):
    rows = []
    cohorts = {"EVAL_V3": "v3", "EVAL_V4": "v4", "EVAL_FAR": "v3"}
    for cohort, prefix in cohorts.items():
        for i in range(4):
            for role in ("A", "B"):
                for qid in (8, 9):
                    noise = 0.002 * ((i + qid) % 3)
                    base = {
                        "cohort": cohort,
                        "scene_id": f"{prefix}_s{i}",
                        "role": role,
                        "query_id": qid,
                        "far_fraction": 0.2,
                    }
                    rows.append(make_row(base, noise, "CONSTANT", 0, 0.5, 0.5))
                    rows.append(make_row(base, noise, "REPROJ_NN", 0, 0.46, harmonic_all + 0.01))
                    rows.append(make_row(base, noise, "REPROJ_HARMONIC", 0, 0.45, harmonic_all))
                    for seed in (1, 2):
                        rows.append(make_row(base, noise, "C0", seed, c0_far, 0.33))
                        rows.append(make_row(base, noise, "C1", seed, c1_far, 0.33))
                        for prefix_method in ("HFILL8", "AHFILL8"):
                            rows.append(
                                make_row(
                                    base,
                                    noise,
                                    f"{prefix_method}_C0",
                                    seed,
                                    c0_far,
                                    hfill_all + 0.01,
                                )
                            )
                            rows.append(
                                make_row(
                                    base, noise, f"{prefix_method}_C1", seed, c1_far, hfill_all
                                )
                            )
    return rows


def test_eval_gates_on_the_pooled_primary_cohorts_set_the_branch():
    analyze_rows = script("evaluate_rgbd_data_scale_eval.py").analyze_rows
    result = analyze_rows(rows_for(0.36, 0.31, 0.28, 0.25))
    assert result["primary"]["n_scenes"] == 8
    assert set(result["cohorts"]) == {"EVAL_V3", "EVAL_V4", "EVAL_FAR"}
    assert result["primary"]["contrasts"]["DATA_GAIN"]["gain"]["mean"] == pytest.approx(0.05)
    assert result["statuses"]["DATA_STATUS"] == "SUPPORTED"
    assert result["statuses"]["HARMONIC_HYBRID_LABEL"] == "ABOVE"
    assert result["INTERPRETATION_BRANCH"] == "A"
    result = analyze_rows(rows_for(0.36, 0.31, 0.24, 0.25))
    assert result["statuses"]["HARMONIC_HYBRID_LABEL"] == "BELOW"
    assert result["INTERPRETATION_BRANCH"] == "B"
    assert analyze_rows(rows_for(0.31, 0.36, 0.24, 0.25))["INTERPRETATION_BRANCH"] == "C"


def test_runner_trains_every_train_set_and_seed_concurrently_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_data_scale_experiment.py")
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
    assert len(seen) == 6 and {a[0] for a in seen} == {"scripts/train_rgbd_data_scale_carrier.py"}
    pairs = {(a[a.index("--variant") + 1], a[a.index("--seed") + 1]) for a in seen}
    assert pairs == {(v, s) for v in ("C0", "C1") for s in ("1", "2", "3")}
    assert all(a[a.index("--device") + 1] == "cpu" for a in seen)
