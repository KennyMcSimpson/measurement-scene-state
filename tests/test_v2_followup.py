"""Held-sequence isolation of the exploratory global-path comparator."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "followup", Path(__file__).resolve().parents[1] / "scripts/explore_v2_discovery.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_rows():
    rows = []
    for s, n in [("a", 2), ("b", 1)]:
        for slot in range(n):
            for path in module.TRAJECTORIES:
                value = 0.0
                if path == ("A", "A"):
                    value = 0.9 if s == "a" else 0.2
                if path == ("B", "B"):
                    value = 0.1 if s == "a" else 0.8
                rows.append({"sequence": s, "target_slot": slot, "trajectory": path, "J": value})
    return rows


def test_held_scores_do_not_choose_held_path():
    rows = fixture_rows()
    before = module.crossfit_global(rows)
    assert all(d["trajectory"] == ["B", "B"] for d in before if d["sequence"] == "a")
    assert all(d["trajectory"] == ["A", "A"] for d in before if d["sequence"] == "b")
    for row in rows:
        if row["sequence"] == "a":
            row["J"] = float(row["trajectory"] == ("OFF", "OFF"))
    after = module.crossfit_global(rows)
    assert [d for d in before if d["sequence"] == "a"] == [d for d in after if d["sequence"] == "a"]
    for d in after:
        assert d["sequence"] not in d["training_sequences"]


def test_missing_or_duplicate_candidate_rejected():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="all16"):
        module.crossfit_global(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        module.crossfit_global(rows + [rows[0]])
