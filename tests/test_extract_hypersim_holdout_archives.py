import importlib.util
import stat
import sys
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "holdout_extraction",
    Path(__file__).resolve().parents[1] / "scripts/extract_hypersim_holdout_archives.py",
)
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def make_archive(root: Path, scene: str = "ai_001_001", payload: bytes = b"rgb-bytes") -> Path:
    archive_path = root / f"{scene}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{scene}/", b"")
        archive.writestr(f"{scene}/rgb/000.png", payload)
    return archive_path


def test_member_paths_reject_absolute_drive_and_parent_traversal() -> None:
    with pytest.raises(ValueError, match="Absolute|drive"):
        extractor._safe_member_parts("C:/outside.txt")
    with pytest.raises(ValueError, match="Absolute|drive"):
        extractor._safe_member_parts("/outside.txt")
    with pytest.raises(ValueError, match="Parent traversal"):
        extractor._safe_member_parts("scene/../outside.txt")


def test_archive_inventory_requires_matching_scene_root_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "ai_001_001.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("other/file.txt", b"wrong root")
    with pytest.raises(ValueError, match="top-level scene mismatch"):
        extractor._inspect_archive(
            "ai_001_001",
            archive_path,
            archive_path.stat().st_size,
            tmp_path / "target",
        )

    symlink_archive = tmp_path / "ai_001_001_symlink.zip"
    info = zipfile.ZipInfo("ai_001_001/link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink_archive, "w") as archive:
        archive.writestr(info, b"target")
    with pytest.raises(ValueError, match="symlink"):
        extractor._inspect_archive(
            "ai_001_001",
            symlink_archive,
            symlink_archive.stat().st_size,
            tmp_path / "target",
        )


def test_extract_member_crc_and_resume_existing_valid_file(tmp_path: Path) -> None:
    scene = "ai_001_001"
    archive_path = make_archive(tmp_path, scene, payload=b"stable-payload")
    target_root = tmp_path / "target"
    target_root.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        info = next(info for info in archive.infolist() if not info.is_dir())
        target = extractor._member_target(target_root, info)
        written = extractor._extract_member(archive, info, target_root, scene)
        assert written == len(b"stable-payload")
        assert target.read_bytes() == b"stable-payload"
        assert extractor._existing_matches(target, info, target_root)

        target.write_bytes(b"corrupted")
        assert not extractor._existing_matches(target, info, target_root)
        extractor._extract_member(archive, info, target_root, scene)
        assert target.read_bytes() == b"stable-payload"
        assert extractor._existing_matches(target, info, target_root)


def test_inspect_archive_reports_member_and_uncompressed_totals(tmp_path: Path) -> None:
    scene = "ai_001_001"
    archive_path = make_archive(tmp_path, scene, payload=b"12345")
    plan = extractor._inspect_archive(
        scene,
        archive_path,
        archive_path.stat().st_size,
        tmp_path / "target",
    )
    assert plan.member_count == 2
    assert plan.uncompressed_bytes == 5
    assert plan.top_level == (scene,)
