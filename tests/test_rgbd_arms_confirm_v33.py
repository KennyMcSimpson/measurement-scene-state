"""V33: the run condition, the branch and which scenes enter the primary pool."""

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_condition_is_a_positive_v32_p1_mean():
    analysis = script("analyze_rgbd_arms_confirm_v33.py")

    def result(gain):
        return {"P1_CONTEXT_GAIN_T24000": {"gain": gain}}

    assert analysis.run_condition(result({"mean": 0.01}))
    assert not analysis.run_condition(result({"mean": 0.0}))
    assert not analysis.run_condition(result({"mean": -0.02}))
    assert not analysis.run_condition(result(None))


def test_branch_needs_both_the_confirmed_gain_and_the_hybrid_label():
    analysis = script("analyze_rgbd_arms_confirm_v33.py")
    assert analysis.branch("SUPPORTED", "ABOVE") == "A"
    assert analysis.branch("SUPPORTED", "NOT_DISTINGUISHABLE") == "B"
    assert analysis.branch("PARTIAL", "ABOVE") == "B"
    assert analysis.branch("NOT_ESTABLISHED", "BELOW") == "C"


def test_primary_pool_is_the_two_hypersim_cohorts_only():
    analysis = script("analyze_rgbd_arms_confirm_v33.py")
    seen = []

    def block(rows, arms, reference):
        seen.append(sorted({r["cohort"] for r in rows}))
        assert arms == ["G1", "P", "U"] and reference == "G1"
        return {
            "contrasts": {
                "G1_MINUS_U_FAR": {"gate": "SUPPORTED"},
                "HARMONIC_MINUS_AHFILL8_U": {"label": "NOT_DISTINGUISHABLE"},
            },
            "n_scenes": len({r["scene_id"] for r in rows}),
        }

    rows = [
        {"cohort": c, "scene_id": f"{c}_{i}"}
        for c in ("EVAL_V5", "TRAIN_EXT", "REPLICA")
        for i in (0, 1)
    ]
    result = analysis.analyze(rows, block)
    assert seen[0] == ["EVAL_V5", "TRAIN_EXT"]
    assert result["hypersim_pooled"]["n_scenes"] == 4
    assert sorted(result["cohorts"]) == ["EVAL_V5", "REPLICA", "TRAIN_EXT"]
    assert result["INTERPRETATION_BRANCH"] == "B"
