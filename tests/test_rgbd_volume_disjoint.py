"""V16 volume-disjoint check: unpaired difference, branch rule, lock guards."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_rgbd_volume_disjoint.py"
spec = importlib.util.spec_from_file_location("evaluate_rgbd_volume_disjoint", SCRIPT)
v16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v16)


def test_unpaired_difference_of_scene_means():
    first = {f"a{i}": 0.10 + 0.001 * i for i in range(20)}
    second = {f"b{i}": 0.02 + 0.001 * i for i in range(10)}
    stats = v16.unpaired(first, second)
    assert stats["mean"] == pytest.approx(0.085)
    assert stats["ci95"][0] > 0 and stats["n_first"] == 20 and stats["n_second"] == 10


def rows(scene_ids, gap, cohort, carrier):
    """Per scene the completion advantage A = gap + 0.01 * ((i % 5) - 2)."""
    out = []
    for i, sid in enumerate(scene_ids):
        advantage = gap + 0.01 * ((i % 5) - 2)
        for role in ("A", "B"):
            for seed, method, far in (
                (0, "REPROJ_HARMONIC", 0.5),
                (0, "REPROJ_NN", 0.55),
                (1, carrier, 0.5 - advantage),
                (2, carrier, 0.5 - advantage),
            ):
                out.append(
                    {
                        "cohort": cohort,
                        "scene_id": sid,
                        "role": role,
                        "query_id": 14,
                        "method": method,
                        "seed": seed,
                        "absrel_ALL": far,
                        "absrel_NEAR": 0.2,
                        "absrel_FAR": far,
                        "far_fraction": 0.2,
                    }
                )
    return out


def branch(fresh_gap, eval_gap):
    fresh = rows([f"ai_{i:03d}_001" for i in range(12)], fresh_gap, "FRESH_V2", "V11C1")
    evals = rows([f"ai_{i:03d}_002" for i in range(40)], eval_gap, "EVAL_V3_NEAR", "C1")
    return v16.analyze_rows(fresh, evals, set())["INTERPRETATION_BRANCH"]


def test_branches_follow_p1_and_the_familiarity_gap():
    assert branch(0.07, 0.07) == "A"
    assert branch(0.03, 0.10) == "B"
    assert branch(-0.02, 0.07) == "C"
    assert branch(0.0, 0.0) == "D"


def test_lock_needs_the_exact_directory_and_a_protocol(tmp_path):
    with pytest.raises(PermissionError):
        v16.lock(tmp_path / "EXP-OTHER")
    root = tmp_path / v16.EXPERIMENT
    root.mkdir()
    with pytest.raises(PermissionError):
        v16.lock(root)
    assert not (root / "lock.json").exists()
