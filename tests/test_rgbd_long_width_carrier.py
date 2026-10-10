"""V19 width-64 long training: variant, V12 C1 draw, checkpoints, control, labels, P2, branch."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mcss.mechanism_pilot import rgbd_data_scale_carrier as v14
from mcss.mechanism_pilot import rgbd_long_width_carrier as v19
from mcss.mechanism_pilot import rgbd_width_carrier as v12

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_variant_is_the_v12_c1_recipe_at_width_64():
    assert v19.VARIANTS == ("L",) and v19.EXPERIMENT == "EXP-3D-RGBD-LONG-TRAIN-WIDTH-V19"
    spec = v19.VARIANT_SPECS["L"]
    assert spec["train"] == "TRAIN72" and (spec["hidden"], spec["expansion"]) == (64, 128)
    assert {k: spec[k] for k in ("hidden", "expansion", "model", "train")} == {
        k: v12.VARIANT_SPECS["C1"][k] for k in ("hidden", "expansion", "model", "train")
    }
    carrier = v19.make_carrier("L", 20260928, "cpu")
    assert sum(p.numel() for p in carrier.parameters()) == 250542


@pytest.mark.parametrize("seed", [20260928, 20260929, 20260930])
def test_parameter_draw_equals_v12_c1(seed):
    mine = v19.parameter_hash(v19.make_carrier("L", seed, "cpu"))
    theirs = v12.parameter_hash(v12.make_carrier("C1", seed, "cpu"))
    assert mine == theirs


def test_checkpoint_roundtrip_and_schema_guard(tmp_path):
    carrier = v19.make_carrier("L", 20260928, "cpu")
    path = tmp_path / "step_000100.pt"
    digest = v19.save_checkpoint(
        path, carrier, variant="L", seed=20260928, step=100, lock_sha256="x"
    )
    assert len(digest) == 64
    restored, payload = v19.load_checkpoint(path, "cpu")
    assert payload["step"] == 100
    assert all(
        torch.equal(a, b)
        for a, b in zip(restored.state_dict().values(), carrier.state_dict().values(), strict=True)
    )
    with pytest.raises(FileExistsError):
        v19.save_checkpoint(path, carrier, variant="L", seed=20260928, step=100, lock_sha256="x")
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
        v19.load_checkpoint(other, "cpu")


def test_control_compares_logs_without_seconds_and_parameters_exactly(tmp_path):
    control = script("check_rgbd_long_width_prefix.py")
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
    carrier = v19.make_carrier("L", 20260928, "cpu")
    p, q = tmp_path / "p.pt", tmp_path / "q.pt"
    v19.save_checkpoint(p, carrier, variant="L", seed=20260928, step=6000, lock_sha256="one")
    v19.save_checkpoint(q, carrier, variant="L", seed=20260928, step=6000, lock_sha256="two")
    assert control.compare_parameters(p, q)["equal"]
    assert control.compare_parameters(tmp_path / "absent.pt", p)["equal"] is False
    assert control.compare_logs(tmp_path / "absent.jsonl", a, steps=6)["equal"] is False
    with torch.no_grad():
        next(carrier.parameters()).add_(1.0)
    r = tmp_path / "r.pt"
    v19.save_checkpoint(r, carrier, variant="L", seed=20260928, step=6000, lock_sha256="one")
    assert control.compare_parameters(p, r)["equal"] is False


def test_labels_cover_the_curve_on_primary_cohorts_only():
    evaluator = script("evaluate_rgbd_long_width_eval.py")
    config = {
        "eval": {
            "curve_steps_primary": [1000, 6000, 24000],
            "steps_descriptive_cohorts": [6000, 24000],
        }
    }
    assert evaluator.labels_for("EVAL_V3", config) == ["T1000", "T6000", "T24000", "SEL"]
    assert evaluator.labels_for("FRESH_V2", config) == ["T6000", "T24000", "SEL"]


def rows(length_gain, far6=0.45, scenes=40, cohorts=("EVAL_V3", "EVAL_V4")):
    """Per scene FAR(T6000) = far6 and FAR(T24000) = far6 - length_gain (with a little jitter)."""
    out = []
    for i in range(scenes):
        jitter = 0.002 * ((i % 5) - 2)
        values = {
            ("REPROJ_NN", 0): (0.32, 0.22, 0.52),
            ("REPROJ_HARMONIC", 0): (0.30, 0.20, 0.50),
            ("AHFILL8_T24000", 1): (0.30, 0.20, 0.45),
            ("T6000", 1): (0.31, 0.21, far6),
            ("T24000", 1): (0.31, 0.21, far6 - length_gain - jitter),
        }
        for (method, seed), (all_, near, far) in values.items():
            for role in ("A", "B"):
                for qid in (14, 15):
                    out.append(
                        {
                            "cohort": cohorts[i % len(cohorts)],
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


def test_analyze_reports_p1_and_leaves_the_branch_to_the_width_contrast():
    evaluator = script("evaluate_rgbd_long_width_eval.py")
    supported = evaluator.analyze_rows(rows(0.03))
    assert supported["statuses"]["LENGTH_STATUS"] == "SUPPORTED"
    assert supported["INTERPRETATION_BRANCH"] == "PENDING_WIDTH_CONTRAST"
    assert evaluator.analyze_rows(rows(0.0))["statuses"]["LENGTH_STATUS"] != "SUPPORTED"


def test_width_contrast_pairs_scenes_and_sets_the_branch():
    width = script("analyze_rgbd_long_width_vs_v17.py")
    # V17 (width 32) gains 0.01 from length; V19 (width 64) gains 0.04 and ends 0.03 lower.
    better = width.width_contrast(rows(0.01), rows(0.04), gate=True)
    gain = better["contrasts"]["WIDTH_GAIN_24K"]
    assert gain["gain"]["mean"] == pytest.approx(0.03, abs=1e-9) and gain["gate"] == "SUPPORTED"
    assert better["contrasts"]["WIDTH_GAIN_6K"]["gain"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert better["contrasts"]["INTERACTION"]["gain"]["mean"] == pytest.approx(-0.03, abs=1e-9)
    same = width.width_contrast(rows(0.02), rows(0.02), gate=True)
    assert same["contrasts"]["WIDTH_GAIN_24K"]["gate"] != "SUPPORTED"
    descriptive = width.width_contrast(rows(0.01), rows(0.04), gate=False)
    assert "gate" not in descriptive["contrasts"]["WIDTH_GAIN_24K"]
    with pytest.raises(PermissionError):
        width.width_contrast(rows(0.01, scenes=40), rows(0.01, scenes=39), gate=True)
    assert width.branch("SUPPORTED", "SUPPORTED") == "A"
    assert width.branch("SUPPORTED", "NOT_ESTABLISHED") == "B"
    assert width.branch("NOT_ESTABLISHED", "SUPPORTED") == "B"
    assert width.branch("SUPPORTED", "NO_DATA") == "B"
    assert width.branch("NOT_ESTABLISHED", "NO_DATA") == "C"


def test_v17_rows_check_the_frozen_v17(tmp_path):
    width = script("analyze_rgbd_long_width_vs_v17.py")
    (tmp_path / "audit").mkdir()
    (tmp_path / "preregistration.json").write_text("{}")
    digest = width.sha(tmp_path / "preregistration.json")
    with pytest.raises(PermissionError):
        width.v17_rows(tmp_path, "0" * 64, poll=0, max_wait=0)
    with pytest.raises(TimeoutError):
        width.v17_rows(tmp_path, digest, poll=0, max_wait=0)
    (tmp_path / "audit/v11_prefix_control.json").write_text(json.dumps({"status": "FAIL"}))
    assert width.v17_rows(tmp_path, digest, poll=0, max_wait=0) == (
        None,
        {"state": "CONTROL_FAILED"},
    )
    (tmp_path / "audit/v11_prefix_control.json").write_text(json.dumps({"status": "PASS"}))
    (tmp_path / "raw/eval").mkdir(parents=True)
    scored = tmp_path / "raw/eval/query_rows.json"
    scored.write_text(json.dumps(rows(0.01, scenes=4)))
    analysis = {
        "control": "PASS",
        "input_sha256": {str(scored): width.sha(scored)},
        "statuses": {"LENGTH_STATUS": "SUPPORTED"},
        "INTERPRETATION_BRANCH": "B",
    }
    (tmp_path / "eval_results.json").write_text(json.dumps(analysis))
    loaded, record = width.v17_rows(tmp_path, digest, poll=0, max_wait=0)
    assert len(loaded) == len(rows(0.01, scenes=4)) and record["state"] == "COMPLETE"
    scored.write_text(json.dumps(rows(0.02, scenes=4)))
    with pytest.raises(PermissionError):
        width.v17_rows(tmp_path, digest, poll=0, max_wait=0)


def test_prepare_needs_the_exact_directory_and_a_protocol(tmp_path):
    prepare = script("prepare_rgbd_long_width_experiment.py")
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
    assert prepare.COUNTS == {"L": 250542}
    assert "EXP-3D-RGBD-LONG-TRAIN-V17" in prepare.CONCURRENT
