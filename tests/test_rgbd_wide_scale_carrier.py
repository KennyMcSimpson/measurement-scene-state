"""V15 width x data: only the width differs on TRAIN72+EXT; V11/V12 draws; gates, 2x2, runner."""

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
from mcss.mechanism_pilot import rgbd_data_scale_carrier as v14
from mcss.mechanism_pilot import rgbd_evidence_carrier as v5
from mcss.mechanism_pilot import rgbd_resolution_carrier as v8
from mcss.mechanism_pilot import rgbd_width_carrier as v12
from mcss.mechanism_pilot.rgbd_wide_scale_carrier import (
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


def test_only_the_width_differs_and_the_draws_are_v11_c1_and_v12_c1():
    assert {v: (s["hidden"], s["expansion"], s["train"]) for v, s in VARIANT_SPECS.items()} == {
        "C0": (32, 64, "TRAIN_EXT"),
        "C1": (64, 128, "TRAIN_EXT"),
    }
    narrow, wide = make_carrier("C0", 3, "cpu"), make_carrier("C1", 3, "cpu")
    assert hash_value(narrow.state_dict()) == hash_value(
        v11.make_carrier("C1", 3, "cpu").state_dict()
    )
    assert hash_value(narrow.state_dict()) == hash_value(
        v14.make_carrier("C1", 3, "cpu").state_dict()
    )
    assert hash_value(wide.state_dict()) == hash_value(
        v12.make_carrier("C1", 3, "cpu").state_dict()
    )
    assert sum(p.numel() for p in narrow.parameters()) == 64238
    assert sum(p.numel() for p in wide.parameters()) == 250542


def test_both_variants_train_on_the_shared_train_ext_set():
    manifest = {"scenes": [{"scene_id": f"t{i:02d}", "split": "TRAIN"} for i in range(30)]}
    config = {
        "train_sets": {
            "TRAIN72": [f"t{i:02d}" for i in range(24)],
            "TRAIN_EXT": [f"t{i:02d}" for i in range(30)],
        }
    }
    for variant in VARIANT_SPECS:
        assert [r["scene_id"] for r in train_records(manifest, config, variant)] == config[
            "train_sets"
        ]["TRAIN_EXT"]


def test_width32_reproduces_the_v14_c1_state_and_loss_exactly():
    rgbd, bounds, cameras = context()
    old, new = v14.make_carrier("C1", 11, "cpu"), make_carrier("C0", 11, "cpu")
    with torch.no_grad():
        for carrier in (old, new):
            carrier.depth_weight.fill_(0.2)
    first, _ = v14.build_state(old, rgbd, bounds, "same")
    second, _ = build_state(new, rgbd, bounds, "same")
    assert hash_scene_state(first) == hash_scene_state(second)
    a, _ = v14.training_loss(first, cameras, *targets(), "C1")
    b, _ = training_loss(second, cameras, *targets(), "C0")
    assert torch.equal(a, b)


def test_the_wide_carrier_trains_and_roundtrips_under_the_v15_schema_only(tmp_path):
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
    assert payload["config"]["hidden_dim"] == 64
    assert hash_value(restored.state_dict()) == hash_value(carrier.state_dict())
    for other in (v12.load_checkpoint, v14.load_checkpoint):
        with pytest.raises(PermissionError):
            other(path, "cpu")
    old = tmp_path / "v12.pt"
    v12.save_checkpoint(
        old, v12.make_carrier("C1", 5, "cpu"), variant="C1", seed=5, step=1, lock_sha256="v12"
    )
    with pytest.raises(PermissionError):
        load_checkpoint(old, "cpu")
    odd = v8.ResolutionRGBDCarrier(
        CarrierConfig(grid_size=(32, 32, 32), token_count=32**3, hidden_dim=16, expansion_dim=32)
    )
    with pytest.raises(ValueError):
        build_state(odd, rgbd, bounds, "odd")


def test_dev_curve_evaluations_do_not_save_states():
    source = (ROOT / "src/mcss/mechanism_pilot/rgbd_wide_scale_evaluation.py").read_text()
    guarded = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "diagnostics"
        and any("states.pt" in ast.unparse(child) for child in node.body)
    ]
    assert len(guarded) == 1
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


def rows_for(c0_far, c1_far, harmonic_all, hfill_all, w32_t72_far=0.40, w64_t72_far=0.40):
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
                        cells = {"C0": c0_far, "C1": c1_far}
                        if cohort != "EVAL_FAR":
                            cells |= {
                                "W32_T72": w32_t72_far,
                                "W64_T72": w64_t72_far,
                                "C0_LAST": c0_far,
                                "C1_LAST": c1_far,
                            }
                        for method, far in cells.items():
                            total = hfill_all if method.startswith("C1") else hfill_all + 0.01
                            rows.append(make_row(base, noise, method, seed, far, 0.33))
                            rows.append(make_row(base, noise, f"HFILL8_{method}", seed, far, total))
                            if method in ("C0", "C1"):
                                rows.append(
                                    make_row(base, noise, f"AHFILL8_{method}", seed, far, total)
                                )
    return rows


def test_eval_gates_and_the_width_by_data_interaction():
    analyze_rows = script("evaluate_rgbd_wide_scale_eval.py").analyze_rows
    result = analyze_rows(rows_for(0.44, 0.39, 0.28, 0.25, w32_t72_far=0.42, w64_t72_far=0.42))
    contrasts = result["primary"]["contrasts"]
    assert result["primary"]["n_scenes"] == 8
    assert contrasts["WIDTH_AT_SCALE_GAIN"]["gain"]["mean"] == pytest.approx(0.05)
    assert contrasts["INTERACTION"]["gain"]["mean"] == pytest.approx(-0.05)
    assert contrasts["DATA_AT_W32_GAIN"]["gain"]["mean"] == pytest.approx(-0.02)
    assert result["cohorts"]["EVAL_FAR"]["contrasts"]["INTERACTION"]["gain"] is None
    assert (
        result["statuses"]["WIDTH_AT_SCALE_STATUS"],
        result["statuses"]["HARMONIC_HYBRID_LABEL"],
    ) == ("SUPPORTED", "ABOVE")
    assert result["INTERPRETATION_BRANCH"] == "A"
    result = analyze_rows(rows_for(0.44, 0.39, 0.24, 0.25))
    assert result["statuses"]["HARMONIC_HYBRID_LABEL"] == "BELOW"
    assert result["INTERPRETATION_BRANCH"] == "B"
    assert analyze_rows(rows_for(0.39, 0.44, 0.24, 0.25))["INTERPRETATION_BRANCH"] == "C"


def test_runner_trains_every_width_and_seed_concurrently_on_cpu(tmp_path, monkeypatch):
    module = script("run_rgbd_wide_scale_experiment.py")
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
    assert len(seen) == 6 and {a[0] for a in seen} == {"scripts/train_rgbd_wide_scale_carrier.py"}
    pairs = {(a[a.index("--variant") + 1], a[a.index("--seed") + 1]) for a in seen}
    assert pairs == {(v, s) for v in ("C0", "C1") for s in ("1", "2", "3")}
