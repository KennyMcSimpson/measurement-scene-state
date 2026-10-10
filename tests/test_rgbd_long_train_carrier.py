"""V17 long training: variants, V11 C1 draw, checkpoints, control, labels and branches."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mcss.mechanism_pilot import rgbd_data_scale_carrier as v14
from mcss.mechanism_pilot import rgbd_long_train_carrier as v17

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_variant_is_the_v11_c1_recipe_on_train72():
    assert v17.VARIANTS == ("L",) and v17.EXPERIMENT == "EXP-3D-RGBD-LONG-TRAIN-V17"
    spec = v17.VARIANT_SPECS["L"]
    assert spec["train"] == "TRAIN72" and (spec["hidden"], spec["expansion"]) == (32, 64)
    carrier = v17.make_carrier("L", 20260928, "cpu")
    assert sum(p.numel() for p in carrier.parameters()) == 64238


@pytest.mark.parametrize("seed", [20260928, 20260929, 20260930])
def test_parameter_draw_equals_v14_c0_which_is_v11_c1(seed):
    mine = v17.parameter_hash(v17.make_carrier("L", seed, "cpu"))
    theirs = v14.parameter_hash(v14.make_carrier("C0", seed, "cpu"))
    assert mine == theirs


def test_checkpoint_roundtrip_and_schema_guard(tmp_path):
    carrier = v17.make_carrier("L", 20260928, "cpu")
    path = tmp_path / "step_000100.pt"
    digest = v17.save_checkpoint(
        path, carrier, variant="L", seed=20260928, step=100, lock_sha256="x"
    )
    assert len(digest) == 64
    restored, payload = v17.load_checkpoint(path, "cpu")
    assert payload["step"] == 100
    assert all(
        torch.equal(a, b)
        for a, b in zip(restored.state_dict().values(), carrier.state_dict().values(), strict=True)
    )
    with pytest.raises(FileExistsError):
        v17.save_checkpoint(path, carrier, variant="L", seed=20260928, step=100, lock_sha256="x")
    other = tmp_path / "v14.pt"
    v14.save_checkpoint(
        other,
        v14.make_carrier("C0", 20260928, "cpu"),
        variant="C0",
        seed=20260928,
        step=1,
        lock_sha256="x",
    )
    with pytest.raises(PermissionError):
        v17.load_checkpoint(other, "cpu")


def test_control_compares_logs_without_seconds_and_parameters_exactly(tmp_path):
    control = script("check_rgbd_long_train_prefix.py")
    rows = [
        {"step": i, "loss": 1.0 / i, "seconds": 0.5, "terms": {"A": {"loss": 0.1}}}
        for i in range(1, 7)
    ]
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    b.write_text("\n".join(json.dumps({**r, "seconds": 9.0}) for r in rows) + "\n")
    changed = [dict(r) for r in rows]
    changed[3]["loss"] += 1e-12
    c.write_text("\n".join(json.dumps(r) for r in changed) + "\n")
    assert control.compare_logs(a, b, steps=6)["equal"]
    assert control.compare_logs(a, c, steps=6) == {"equal": False, "first_mismatch_step": 4}
    assert control.compare_logs(a, b, steps=7)["equal"] is False
    carrier = v17.make_carrier("L", 20260928, "cpu")
    p, q = tmp_path / "p.pt", tmp_path / "q.pt"
    v17.save_checkpoint(p, carrier, variant="L", seed=20260928, step=6000, lock_sha256="one")
    v17.save_checkpoint(q, carrier, variant="L", seed=20260928, step=6000, lock_sha256="two")
    assert control.compare_parameters(p, q)["equal"]
    assert control.compare_parameters(tmp_path / "absent.pt", p)["equal"] is False
    assert control.compare_logs(tmp_path / "absent.jsonl", a, steps=6)["equal"] is False
    with torch.no_grad():
        next(carrier.parameters()).add_(1.0)
    r = tmp_path / "r.pt"
    v17.save_checkpoint(r, carrier, variant="L", seed=20260928, step=6000, lock_sha256="one")
    assert control.compare_parameters(p, r)["equal"] is False


def test_labels_cover_the_curve_on_primary_cohorts_only():
    evaluator = script("evaluate_rgbd_long_train_eval.py")
    config = {
        "eval": {
            "curve_steps_primary": [1000, 6000, 24000],
            "steps_descriptive_cohorts": [6000, 24000],
        }
    }
    assert evaluator.labels_for("EVAL_V3", config) == ["T1000", "T6000", "T24000", "SEL"]
    assert evaluator.labels_for("FRESH_V2", config) == ["T6000", "T24000", "SEL"]


def rows(length_gain, hybrid_gain, scenes=40):
    """FAR(T6000) - FAR(T24000) = length_gain; harmonic - AHFILL8 = hybrid_gain (per scene)."""
    out = []
    for i in range(scenes):
        jitter = 0.002 * ((i % 5) - 2)
        cohort = "EVAL_V3" if i % 2 else "EVAL_V4"
        values = {
            ("REPROJ_NN", 0): (0.32, 0.22, 0.52),
            ("REPROJ_HARMONIC", 0): (0.30, 0.20, 0.50),
            ("AHFILL8_T24000", 1): (0.30 - hybrid_gain - jitter, 0.20, 0.45),
            ("T6000", 1): (0.31, 0.21, 0.45),
            ("T24000", 1): (0.31, 0.21, 0.45 - length_gain - jitter),
        }
        for (method, seed), (all_, near, far) in values.items():
            for role in ("A", "B"):
                for qid in (14, 15):
                    out.append(
                        {
                            "cohort": cohort,
                            "scene_id": f"ai_{i:03d}_001",
                            "role": role,
                            "query_id": qid,
                            "method": method,
                            "seed": seed,
                            "absrel_ALL": all_,
                            "absrel_NEAR": near,
                            "absrel_FAR": far,
                            "far_fraction": 0.2,
                        }
                    )
    return out


def test_branches_follow_p1_gate_and_p2_label():
    evaluator = script("evaluate_rgbd_long_train_eval.py")
    branch = lambda a, b: evaluator.analyze_rows(rows(a, b))["INTERPRETATION_BRANCH"]  # noqa: E731
    assert branch(0.03, 0.03) == "A"
    assert branch(0.03, 0.0) == "B"
    assert branch(0.0, 0.03) == "B"
    assert branch(0.0, 0.0) == "C"


def test_prepare_needs_the_exact_directory_and_a_protocol(tmp_path):
    prepare = script("prepare_rgbd_long_train_experiment.py")
    with pytest.raises(PermissionError):
        prepare.prepare(tmp_path / "EXP-OTHER")
    root = tmp_path / prepare.EXPERIMENT
    root.mkdir()
    with pytest.raises(PermissionError):
        prepare.prepare(root)
    assert not (root / "preregistration.json").exists()
    assert (
        prepare.CHECKPOINT_STEPS[:12] == prepare.V11_CHECKPOINTS
        and prepare.CHECKPOINT_STEPS[-1] == 24000
    )
