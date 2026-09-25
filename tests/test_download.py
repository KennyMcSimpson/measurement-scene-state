from pathlib import Path

import pytest

import mcss.data.download as download
from mcss.data.download import (
    build_hypersim_plan,
    build_replica_plan,
    hypersim_scene_url,
    replica_part_urls,
    safe_member_destination,
)


def test_official_replica_plan_contains_all_seventeen_parts(tmp_path: Path) -> None:
    urls = replica_part_urls()
    plan = build_replica_plan(tmp_path)

    assert len(urls) == 17
    assert urls[0].endswith("replica_v1_0.tar.gz.partaa")
    assert urls[-1].endswith("replica_v1_0.tar.gz.partaq")
    assert len(plan.items) == 17
    assert all(item.destination.parent == tmp_path / "archives" for item in plan.items)


def test_hypersim_subset_plan_is_scene_bounded(tmp_path: Path) -> None:
    plan = build_hypersim_plan(tmp_path, ["ai_001_001", "ai_001_002"])

    assert len(plan.items) == 2
    assert plan.items[0].url == hypersim_scene_url("ai_001_001")
    assert plan.dataset == "hypersim"
    with pytest.raises(ValueError, match="scene"):
        hypersim_scene_url("../escape")


def test_archive_members_cannot_escape_destination(tmp_path: Path) -> None:
    destination = safe_member_destination(tmp_path, "scene/rgb/frame.jpg")
    assert destination == tmp_path / "scene" / "rgb" / "frame.jpg"

    with pytest.raises(ValueError, match="unsafe"):
        safe_member_destination(tmp_path, "../outside.txt")
    with pytest.raises(ValueError, match="unsafe"):
        safe_member_destination(tmp_path, "C:/outside.txt")


def test_hypersim_official_scene_prefix_is_normalized_once(tmp_path: Path) -> None:
    rooted_name = "ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5"

    relative_name = download.hypersim_member_relative_name("ai_001_001", rooted_name)

    assert relative_name == ("images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5")
    assert download._select_hypersim_member(
        relative_name,
        camera="cam_00",
        frame_ids=None,
        include_position=False,
    )
    assert safe_member_destination(tmp_path / "ai_001_001", relative_name) == (
        tmp_path
        / "ai_001_001"
        / "images"
        / "scene_cam_00_geometry_hdf5"
        / "frame.0000.depth_meters.hdf5"
    )
    with pytest.raises(ValueError, match="different scene"):
        download.hypersim_member_relative_name("ai_001_001", rooted_name.replace("001", "002", 1))
