"""Safely extract the approved Hypersim final-holdout archives.

The archive bytes and ZIP central directories are treated as input metadata.  Each
member is extracted with an explicit path and CRC check; ``ZipFile.extractall`` is
intentionally not used.  The operation is resumable: an existing regular file is
validated before it is skipped, and incomplete writes use a same-directory temp
file followed by an atomic rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import time
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mcss.file_lock import acquire_file_lock, release_file_lock

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs/dataset_access_check_20260918/hypersim_holdout_download_sizes.json"
ARCHIVE_ROOT = PROJECT / "data/hypersim_final_holdout_archives"
TARGET_ROOT = PROJECT / "data/hypersim_final_holdout_raw"
LOG_ROOT = PROJECT / "outputs/hypersim_holdout_extraction_20260919"
RESERVE_BYTES = 10 * 1024**3
COPY_CHUNK_BYTES = 8 * 1024**2
SCENE_RE = re.compile(r"^ai_\d{3}_\d{3}$")
DRIVE_RE = re.compile(r"^[A-Za-z]:")
REPARSE_POINT = 0x0400


@dataclass(frozen=True)
class ArchivePlan:
    scene: str
    path: Path
    expected_bytes: int
    archive_bytes: int
    infos: tuple[zipfile.ZipInfo, ...]
    member_count: int
    uncompressed_bytes: int
    top_level: tuple[str, ...]


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {description}: expected an object: {path}")
    return value


def _is_reparse(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(mode) or bool(attributes & REPARSE_POINT)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_project_child(path: Path, parent: Path, description: str) -> Path:
    parent_resolved = parent.resolve()
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(parent_resolved)
    except ValueError as error:
        raise ValueError(f"{description} escapes its allowed root: {path}") from error
    if not relative.parts:
        raise ValueError(f"{description} may not be the allowed root: {path}")
    return resolved


def _assert_safe_chain(root: Path, path: Path) -> None:
    root_resolved = root.resolve()
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"Path escapes extraction root: {path}") from error
    current = root_resolved
    if _is_reparse(current):
        raise ValueError(f"Extraction root is a symlink or reparse point: {current}")
    for part in relative.parts:
        current /= part
        if _lexists(current) and _is_reparse(current):
            raise ValueError(f"Path contains a symlink or reparse point: {current}")


def _ensure_directory(path: Path, root: Path) -> None:
    _assert_safe_chain(root, path)
    if _lexists(path):
        if not path.is_dir():
            raise ValueError(f"Expected directory, found non-directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)
    _assert_safe_chain(root, path)


def _safe_member_parts(name: str) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError(f"Invalid ZIP member name: {name!r}")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or DRIVE_RE.match(normalized):
        raise ValueError(f"Absolute or drive-qualified ZIP member: {name!r}")
    parts = PurePosixPath(normalized).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"Parent traversal in ZIP member: {name!r}")
    safe_parts = tuple(part for part in parts if part not in ("", "."))
    if not safe_parts:
        raise ValueError(f"Empty ZIP member path: {name!r}")
    return safe_parts


def _member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _member_target(target_root: Path, info: zipfile.ZipInfo) -> Path:
    parts = _safe_member_parts(info.filename)
    target = target_root.joinpath(*parts)
    _assert_safe_chain(target_root, target)
    return target


def _load_rows(source: Path) -> tuple[list[dict[str, Any]], str]:
    source_bytes = source.read_bytes()
    payload = json.loads(source_bytes)
    if not isinstance(payload, dict):
        raise ValueError("Approved manifest must be a JSON object")
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != 26:
        raise ValueError("Expected exactly 26 approved archive rows")
    if payload.get("verified_scenes") != 26:
        raise ValueError("Approved manifest does not contain 26 verified scenes")
    scenes: list[str] = []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Approved manifest row is not an object")
        scene = row.get("scene")
        if not isinstance(scene, str) or not SCENE_RE.fullmatch(scene):
            raise ValueError(f"Unsafe scene ID in approved manifest: {scene!r}")
        if scene in scenes:
            raise ValueError(f"Duplicate scene in approved manifest: {scene}")
        if row.get("status") != "index_read_ok":
            raise ValueError(f"Scene was not index-verified: {scene}")
        try:
            expected = int(row["zip_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid expected archive size: {scene}") from error
        if expected <= 0:
            raise ValueError(f"Non-positive expected archive size: {scene}")
        scenes.append(scene)
        normalized.append({**row, "zip_bytes": expected})
    return normalized, hashlib.sha256(source_bytes).hexdigest()


def _check_archive_directory(archive_root: Path, scenes: set[str]) -> None:
    _assert_project_child(archive_root, PROJECT / "data", "Archive root")
    if _is_reparse(archive_root):
        raise ValueError(f"Archive root is a symlink or reparse point: {archive_root}")
    entries = list(archive_root.iterdir()) if archive_root.exists() else []
    expected = {f"{scene}.zip" for scene in scenes}
    actual = {entry.name for entry in entries}
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(f"Archive directory mismatch; extra={extra}, missing={missing}")
    for entry in entries:
        if not entry.is_file() or _is_reparse(entry):
            raise ValueError(f"Archive entry is not a regular file: {entry}")


def _inspect_archive(
    scene: str, archive_path: Path, expected_bytes: int, target_root: Path
) -> ArchivePlan:
    actual_bytes = archive_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Archive byte count mismatch for {scene}: {actual_bytes} != {expected_bytes}"
        )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = tuple(archive.infolist())
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Unreadable ZIP archive: {archive_path}") from error
    if not infos:
        raise ValueError(f"Empty ZIP archive: {archive_path}")
    names = [info.filename for info in infos]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate ZIP member in {archive_path}")
    top_level: set[str] = set()
    uncompressed = 0
    for info in infos:
        if _member_is_symlink(info):
            raise ValueError(f"ZIP symlink member is forbidden: {scene}/{info.filename}")
        parts = _safe_member_parts(info.filename)
        top_level.add(parts[0])
        _member_target(target_root, info)
        uncompressed += info.file_size
    if top_level != {scene}:
        raise ValueError(f"ZIP top-level scene mismatch for {scene}: {sorted(top_level)}")
    return ArchivePlan(
        scene=scene,
        path=archive_path,
        expected_bytes=expected_bytes,
        archive_bytes=actual_bytes,
        infos=infos,
        member_count=len(infos),
        uncompressed_bytes=uncompressed,
        top_level=tuple(sorted(top_level)),
    )


def _existing_matches(path: Path, info: zipfile.ZipInfo, target_root: Path) -> bool:
    if not _lexists(path):
        return False
    _assert_safe_chain(target_root, path)
    if not path.is_file():
        raise ValueError(f"Expected regular file for ZIP member: {path}")
    if path.stat().st_size != info.file_size:
        return False
    crc = 0
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
    return size == info.file_size and (crc & 0xFFFFFFFF) == info.CRC


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    target_root: Path,
    scene: str,
) -> int:
    target = _member_target(target_root, info)
    if info.is_dir():
        _ensure_directory(target, target_root)
        return 0
    _ensure_directory(target.parent, target_root)
    temporary = target.with_name(f".{target.name}.mcss-{scene}.part")
    _assert_safe_chain(target_root, temporary)
    if _lexists(temporary):
        if _is_reparse(temporary):
            raise ValueError(
                "Temporary extraction path is a symlink or reparse point: "
                f"{temporary}"
            )
        if not temporary.is_file():
            raise ValueError(f"Temporary extraction path is not a file: {temporary}")
        temporary.unlink()
    crc = 0
    written = 0
    try:
        with archive.open(info, "r") as source, temporary.open("xb") as destination:
            while True:
                chunk = source.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                destination.write(chunk)
                written += len(chunk)
                crc = zlib.crc32(chunk, crc)
            destination.flush()
            os.fsync(destination.fileno())
        if written != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
            raise ValueError(
                f"CRC/size mismatch for {scene}/{info.filename}: "
                f"size={written}/{info.file_size}, crc={crc & 0xFFFFFFFF}/{info.CRC}"
            )
        _assert_safe_chain(target_root, target)
        os.replace(temporary, target)
        return written
    except Exception:
        if _lexists(temporary) and not _is_reparse(temporary):
            temporary.unlink()
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if _is_reparse(path):
        raise ValueError(f"Output path is a symlink or reparse point: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if _is_reparse(temporary):
        raise ValueError(f"Temporary output path is a symlink or reparse point: {temporary}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            acquire_file_lock(handle)
        except OSError as error:
            raise RuntimeError(f"Another extraction process holds the lock: {path}") from error
        try:
            yield
        finally:
            handle.seek(0)
            release_file_lock(handle)
    finally:
        handle.close()


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        handle.flush()


def _target_root(target: Path, scenes: set[str]) -> None:
    _assert_project_child(target, PROJECT / "data", "Target root")
    if _lexists(target):
        if _is_reparse(target) or not target.is_dir():
            raise ValueError(f"Target root is not a normal directory: {target}")
    else:
        target.mkdir(parents=True, exist_ok=True)
    _assert_safe_chain(PROJECT / "data", target)
    unexpected = sorted(child.name for child in target.iterdir() if child.name not in scenes)
    if unexpected:
        raise ValueError(f"Target contains unexpected top-level entries: {unexpected}")


def _estimate_required_bytes(plans: list[ArchivePlan], target_root: Path) -> int:
    required = 0
    for plan in plans:
        for info in plan.infos:
            if info.is_dir():
                continue
            target = _member_target(target_root, info)
            if not _existing_matches(target, info, target_root):
                required += info.file_size
    return required


def _initial_state(
    plans: list[ArchivePlan],
    source: Path,
    source_sha256: str,
    target_root: Path,
    reserve_bytes: int,
    required_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": "mcss.hypersim.holdout.extraction.v1",
        "status": "preflight",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "updated_unix": time.time(),
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": source_sha256,
        "archive_root": str(ARCHIVE_ROOT.resolve()),
        "target_root": str(target_root.resolve()),
        "reserve_bytes": reserve_bytes,
        "required_bytes_after_resume": required_bytes,
        "expected_members": sum(plan.member_count for plan in plans),
        "expected_uncompressed_bytes": sum(plan.uncompressed_bytes for plan in plans),
        "extracted_members": 0,
        "skipped_members": 0,
        "extracted_bytes": 0,
        "skipped_bytes": 0,
        "extraction_performed": False,
        "scientific_evaluation_performed": False,
        "archives": [
            {
                "scene": plan.scene,
                "archive_bytes": plan.archive_bytes,
                "expected_archive_bytes": plan.expected_bytes,
                "members": plan.member_count,
                "uncompressed_bytes": plan.uncompressed_bytes,
                "status": "queued",
                "processed_members": 0,
                "extracted_members": 0,
                "skipped_members": 0,
                "extracted_bytes": 0,
                "skipped_bytes": 0,
            }
            for plan in plans
        ],
    }


def run(
    *,
    source: Path = SOURCE,
    archive_root: Path = ARCHIVE_ROOT,
    target_root: Path = TARGET_ROOT,
    log_root: Path = LOG_ROOT,
    reserve_bytes: int = RESERVE_BYTES,
    dry_run: bool = False,
) -> dict[str, Any]:
    rows, source_sha256 = _load_rows(source)
    scenes = {str(row["scene"]) for row in rows}
    _check_archive_directory(archive_root, scenes)
    _target_root(target_root, scenes)
    plans = [
        _inspect_archive(
            str(row["scene"]),
            archive_root / f"{row['scene']}.zip",
            int(row["zip_bytes"]),
            target_root,
        )
        for row in rows
    ]
    inventory = {
        "schema": "mcss.hypersim.holdout.extraction.inventory.v1",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": source_sha256,
        "target_root": str(target_root.resolve()),
        "archives": [
            {
                "scene": plan.scene,
                "archive_bytes": plan.archive_bytes,
                "expected_archive_bytes": plan.expected_bytes,
                "members": plan.member_count,
                "uncompressed_bytes": plan.uncompressed_bytes,
                "top_level": list(plan.top_level),
            }
            for plan in plans
        ],
        "total_members": sum(plan.member_count for plan in plans),
        "total_uncompressed_bytes": sum(plan.uncompressed_bytes for plan in plans),
        "extraction_performed": False,
        "scientific_evaluation_performed": False,
    }
    log_root_resolved = _assert_project_child(log_root, PROJECT / "outputs", "Log root")
    if _is_reparse(log_root_resolved):
        raise ValueError(f"Log root is a symlink or reparse point: {log_root_resolved}")
    log_root_resolved.mkdir(parents=True, exist_ok=True)
    _atomic_json(log_root_resolved / "inventory.json", inventory)
    required_bytes = _estimate_required_bytes(plans, target_root)
    free_bytes = shutil.disk_usage(target_root.parent).free
    state = _initial_state(
        plans,
        source,
        source_sha256,
        target_root,
        reserve_bytes,
        required_bytes,
    )
    state["free_bytes_at_preflight"] = free_bytes
    state["space_sufficient"] = free_bytes >= required_bytes + reserve_bytes
    _atomic_json(log_root_resolved / "status.json", state)
    if not state["space_sufficient"]:
        state["status"] = "failed"
        state["error"] = (
            f"Insufficient disk space: free={free_bytes}, "
            f"required={required_bytes}, reserve={reserve_bytes}"
        )
        state["updated_unix"] = time.time()
        _atomic_json(log_root_resolved / "status.json", state)
        raise OSError(state["error"])
    if dry_run:
        state["status"] = "dry_run"
        state["updated_unix"] = time.time()
        _atomic_json(log_root_resolved / "status.json", state)
        return state

    progress_path = log_root_resolved / "progress.jsonl"
    with _process_lock(log_root_resolved / "extract.lock"):
        state["status"] = "running"
        state["updated_unix"] = time.time()
        _atomic_json(log_root_resolved / "status.json", state)
        _append_event(progress_path, {"event": "start", "pid": os.getpid()})
        last_status = time.monotonic()
        for archive_index, plan in enumerate(plans):
            archive_state = state["archives"][archive_index]
            archive_state["status"] = "running"
            _append_event(
                progress_path,
                {
                    "event": "archive_start",
                    "scene": plan.scene,
                    "members": plan.member_count,
                    "uncompressed_bytes": plan.uncompressed_bytes,
                },
            )
            try:
                with zipfile.ZipFile(plan.path) as archive:
                    for info in plan.infos:
                        target = _member_target(target_root, info)
                        archive_state["processed_members"] += 1
                        if info.is_dir():
                            _ensure_directory(target, target_root)
                        elif _existing_matches(target, info, target_root):
                            archive_state["skipped_members"] += 1
                            archive_state["skipped_bytes"] += info.file_size
                            state["skipped_members"] += 1
                            state["skipped_bytes"] += info.file_size
                        else:
                            written = _extract_member(archive, info, target_root, plan.scene)
                            archive_state["extracted_members"] += 1
                            archive_state["extracted_bytes"] += written
                            state["extracted_members"] += 1
                            state["extracted_bytes"] += written
                        if time.monotonic() - last_status >= 5:
                            state["updated_unix"] = time.time()
                            _atomic_json(log_root_resolved / "status.json", state)
                            last_status = time.monotonic()
            except Exception as error:
                archive_state["status"] = "failed"
                archive_state["error"] = str(error)
                state["status"] = "failed"
                state["error"] = f"{plan.scene}: {error}"
                state["updated_unix"] = time.time()
                _atomic_json(log_root_resolved / "status.json", state)
                _append_event(
                    progress_path,
                    {"event": "archive_failed", "scene": plan.scene, "error": str(error)},
                )
                raise
            archive_state["status"] = "complete"
            _append_event(progress_path, {"event": "archive_complete", "scene": plan.scene})
            state["updated_unix"] = time.time()
            _atomic_json(log_root_resolved / "status.json", state)
        expected_scenes = scenes
        actual_scenes = {child.name for child in target_root.iterdir() if child.is_dir()}
        if actual_scenes != expected_scenes:
            raise ValueError(
                f"Extracted scene roots mismatch; extra={sorted(actual_scenes - expected_scenes)}, "
                f"missing={sorted(expected_scenes - actual_scenes)}"
            )
        state["status"] = "complete"
        state["extraction_performed"] = True
        state["updated_unix"] = time.time()
        _atomic_json(log_root_resolved / "status.json", state)
        receipt = {
            "schema": "mcss.hypersim.holdout.extraction.receipt.v1",
            "status": "complete",
            "source_manifest": str(source.resolve()),
            "source_manifest_sha256": source_sha256,
            "archive_root": str(archive_root.resolve()),
            "target_root": str(target_root.resolve()),
            "scene_count": len(plans),
            "member_count": state["expected_members"],
            "uncompressed_bytes": state["expected_uncompressed_bytes"],
            "extracted_members": state["extracted_members"],
            "skipped_members": state["skipped_members"],
            "extracted_bytes": state["extracted_bytes"],
            "skipped_bytes": state["skipped_bytes"],
            "crc_verified": True,
            "extraction_performed": True,
            "scientific_evaluation_performed": False,
            "archives": state["archives"],
        }
        _atomic_json(log_root_resolved / "receipt.json", receipt)
        _append_event(progress_path, {"event": "complete", "scene_count": len(plans)})
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT)
    parser.add_argument("--reserve-gib", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            source=args.source,
            archive_root=args.archive_root,
            target_root=args.target_root,
            log_root=args.log_root,
            reserve_bytes=int(args.reserve_gib * 1024**3),
            dry_run=args.dry_run,
        )
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}), flush=True)
        return 1
    print(
        json.dumps({"status": result["status"], "target_root": result["target_root"]}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
