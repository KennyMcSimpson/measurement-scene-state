"""V13 abstention: the fallback rule, the preregistered branches and the lock guards."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_rgbd_volume_abstention.py"
spec = importlib.util.spec_from_file_location("evaluate_rgbd_volume_abstention", SCRIPT)
v13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v13)


def test_protocol_constants_are_frozen():
    assert v13.THRESHOLD == 0.5 and v13.DESCRIPTIVE_THRESHOLDS == (0.3, 0.7, 0.9)
    assert v13.PRIMARY == "EVAL_FAR" and v13.COHORTS["EVAL_FAR"][1:] == ("far_query", "EVAL_FAR")
    assert v13.CONTRASTS["ABSTAIN_GAIN"] == ("HFILL8", "ALL", "AHFILL8", "ALL")
    assert v13.CONTRASTS["HARMONIC_VS_AHFILL8"] == ("REPROJ_HARMONIC", "ALL", "AHFILL8", "ALL")


def test_abstain_takes_harmonic_on_near_and_low_opacity_far_pixels():
    near = np.array([[True, False, False, False]])
    opacity = np.array([[0.0, 0.2, 0.5, 0.9]])
    out = v13.abstain(near, opacity, np.full((1, 4), 1.0), np.full((1, 4), 2.0), 0.5)
    assert out.tolist() == [[1.0, 1.0, 2.0, 2.0]]


def rows(levels, scenes=20):
    out = []
    for s in range(scenes):
        for method, level in levels.items():
            for seed in (0,) if method.startswith("REPROJ") else (1, 2, 3):
                for role in ("A", "B"):
                    value = level + 0.001 * (s % 5)
                    out.append(
                        {
                            "cohort": "EVAL_FAR",
                            "scene_id": f"s{s:02d}",
                            "role": role,
                            "query_id": 32,
                            "method": method,
                            "seed": seed,
                            **{f"absrel_{region}": value for region in v13.REGIONS},
                        }
                    )
    return out


FLAGS = [
    {
        "cohort": "EVAL_FAR",
        "scene_id": f"s{s:02d}",
        "surface_in_volume": 0.5,
        "far_fraction": 0.4,
        "flagged_share_of_far": 0.3,
        "flagged": 3.0,
        "flagged_outside": 2.0,
        "far_outside": 4.0,
    }
    for s in range(20)
]


def test_branches_follow_the_two_primary_checks():
    a = v13.analyze_rows(rows({"HFILL8": 0.30, "AHFILL8": 0.25, "REPROJ_HARMONIC": 0.28}), FLAGS)
    assert (a["ABSTAIN_STATUS"], a["HARMONIC_LABEL"], a["INTERPRETATION_BRANCH"]) == (
        "SUPPORTED",
        "ABOVE",
        "A",
    )
    coverage = a["cohorts"]["EVAL_FAR"]["coverage"]
    assert coverage["precision_outside_volume"] == pytest.approx(2 / 3)
    assert coverage["recall_outside_volume"] == pytest.approx(0.5)
    b = v13.analyze_rows(rows({"HFILL8": 0.25, "AHFILL8": 0.25, "REPROJ_HARMONIC": 0.28}), FLAGS)
    assert (b["ABSTAIN_STATUS"], b["HARMONIC_LABEL"], b["INTERPRETATION_BRANCH"]) == (
        "NOT_ESTABLISHED",
        "ABOVE",
        "B",
    )
    c = v13.analyze_rows(rows({"HFILL8": 0.30, "AHFILL8": 0.30, "REPROJ_HARMONIC": 0.28}), FLAGS)
    assert (c["HARMONIC_LABEL"], c["INTERPRETATION_BRANCH"]) == ("BELOW", "C")


def test_lock_needs_the_exact_directory_a_protocol_and_no_prediction(tmp_path):
    with pytest.raises(PermissionError):
        v13.lock(tmp_path / "EXP-SOMETHING-ELSE")
    root = tmp_path / v13.EXPERIMENT
    root.mkdir()
    with pytest.raises(PermissionError):
        v13.lock(root)
    (root / "PROTOCOL.md").write_text("protocol")
    (root / "raw").mkdir()
    with pytest.raises(PermissionError):
        v13.lock(root)
    assert not (root / "lock.json").exists()
