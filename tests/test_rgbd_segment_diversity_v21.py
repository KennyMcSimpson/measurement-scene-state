"""V21 segment diversity: the branch rule, the paired P1 contrast and the segment data rule."""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(far24, scenes=40, hybrid=0.0):
    """Per scene FAR(T6000) = 0.45 and FAR(T24000) = far24 (with a little jitter)."""
    out = []
    for i in range(scenes):
        jitter = 0.002 * ((i % 5) - 2)
        values = {
            ("REPROJ_NN", 0): (0.32, 0.22, 0.52),
            ("REPROJ_HARMONIC", 0): (0.30, 0.20, 0.50),
            ("AHFILL8_T24000", 1): (0.30 - hybrid - jitter, 0.20, 0.45),
            ("T6000", 1): (0.31, 0.21, 0.45),
            ("T24000", 1): (0.31, 0.21, far24 + jitter),
        }
        for (method, seed), (all_, near, far) in values.items():
            for role in ("A", "B"):
                for qid in (14, 15):
                    out.append(
                        {
                            "cohort": "EVAL_V3" if i % 2 else "EVAL_V4",
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


def test_branch_rule():
    v21 = script("analyze_rgbd_segment_diversity_v21.py")
    assert v21.branch("SUPPORTED", "ABOVE") == "A"
    assert v21.branch("SUPPORTED", "NOT_DISTINGUISHABLE") == "B"
    assert v21.branch("PARTIAL", "ABOVE") == "B"
    assert v21.branch("PARTIAL", "NOT_DISTINGUISHABLE") == "C"


def test_p1_pairs_the_arms_by_scene_and_sets_the_branch():
    v21 = script("analyze_rgbd_segment_diversity_v21.py")
    evaluation = v21.frozen_v17_evaluation()
    g1, g5 = rows(0.40), rows(0.37, hybrid=0.03)
    result = v21.analyze({"G1": g1, "G5": g5}, g1, evaluation)
    p1 = result["P1_DIVERSITY_GAIN_T24000"]
    assert p1["gain"]["mean"] == pytest.approx(0.03, abs=1e-9)
    assert result["statuses"]["DIVERSITY_STATUS"] == "SUPPORTED"
    assert result["INTERPRETATION_BRANCH"] in ("A", "B")
    same = v21.analyze({"G1": g1, "G5": rows(0.40)}, g1, evaluation)
    assert same["statuses"]["DIVERSITY_STATUS"] != "SUPPORTED"
    assert same["descriptive"]["DEVICE_CHECK_V17_CPU_MINUS_G1_GPU_T24000"]["gain"]["mean"] == 0


def test_segment_rule_takes_later_positions_and_never_the_original_frames():
    prep = script("prepare_rgbd_segment_data.py")
    assert prep.SEGMENTS == (1, 2, 3, 4) and prep.SEGMENT_LENGTH == 16
    assert prep.segment_id("ai_001_002", 3) == "ai_001_002_seg3"

    class Info:
        def __init__(self, name):
            self.filename = name

    scene = "ai_001_002"
    names = [f"{scene}/_detail/metadata_scene.csv"] + [
        f"{scene}/_detail/cam_00/camera_keyframe_{k}.hdf5"
        for k in ("frame_indices", "positions", "orientations")
    ]
    for i in range(40):
        names += [
            f"{scene}/images/scene_cam_00_final_preview/frame.{i:04d}.color.jpg",
            f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{i:04d}.depth_meters.hdf5",
            f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{i:04d}.position.hdf5",
        ]
    _, meta, segments = prep.members_of(scene, [Info(n) for n in names], list(range(40)))
    assert len(meta) == 4 and sorted(segments) == [1]
    ids, members = segments[1]
    assert ids == list(range(16, 32)) and len(members) == 33
