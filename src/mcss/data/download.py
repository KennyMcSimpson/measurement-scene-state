"""Safe, resumable download helpers for official Replica and Hypersim releases."""

from __future__ import annotations

import io
import os
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from string import ascii_lowercase
from typing import BinaryIO

import requests

REPLICA_RELEASE_URL = "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"
HYPERSIM_SCENE_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"
)
_SCENE_NAME = re.compile(r"^ai_\d{3}_\d{3}$")


@dataclass(frozen=True)
class DownloadItem:
    url: str
    destination: Path


@dataclass(frozen=True)
class DownloadPlan:
    dataset: str
    root: Path
    items: tuple[DownloadItem, ...]


@dataclass(frozen=True)
class DownloadResult:
    item: DownloadItem
    bytes_written: int
    reused: bool


def replica_part_urls() -> tuple[str, ...]:
    """Return the 17 official v1.0 Replica archive part URLs, aa through aq."""

    return tuple(
        f"{REPLICA_RELEASE_URL}/replica_v1_0.tar.gz.parta{suffix}"
        for suffix in ascii_lowercase[:17]
    )


def build_replica_plan(root: str | Path) -> DownloadPlan:
    root_path = Path(root)
    archive_root = root_path / "archives"
    items = tuple(DownloadItem(url, archive_root / Path(url).name) for url in replica_part_urls())
    return DownloadPlan("replica", root_path, items)


def hypersim_scene_url(scene: str) -> str:
    if not _SCENE_NAME.fullmatch(scene):
        raise ValueError("scene must match ai_DDD_DDD, for example ai_001_001")
    return f"{HYPERSIM_SCENE_URL}/{scene}.zip"


def build_hypersim_plan(root: str | Path, scenes: list[str] | tuple[str, ...]) -> DownloadPlan:
    root_path = Path(root)
    if not scenes:
        raise ValueError("at least one Hypersim scene is required")
    items = tuple(
        DownloadItem(hypersim_scene_url(scene), root_path / "archives" / f"{scene}.zip")
        for scene in scenes
    )
    return DownloadPlan("hypersim", root_path, items)


def safe_member_destination(root: str | Path, member_name: str) -> Path:
    """Map an archive member to a descendant path or reject traversal and drive paths."""

    normalized = member_name.replace("\\", "/")
    pure_path = PurePosixPath(normalized)
    if (
        not normalized
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or re.match(r"^[A-Za-z]:", normalized) is not None
    ):
        raise ValueError(f"unsafe archive member path: {member_name!r}")
    base = Path(root).resolve()
    destination = (base / Path(*pure_path.parts)).resolve()
    try:
        destination.relative_to(base)
    except ValueError as error:
        raise ValueError(f"unsafe archive member path: {member_name!r}") from error
    return destination


def execute_plan(
    plan: DownloadPlan,
    *,
    dry_run: bool = False,
    session: requests.Session | None = None,
    timeout_seconds: float = 60.0,
) -> list[DownloadResult]:
    """Download all plan items with `.part` resume files and atomic final renames."""

    if dry_run:
        return [DownloadResult(item, 0, False) for item in plan.items]
    active_session = session or requests.Session()
    owns_session = session is None
    try:
        return [
            download_file(item, session=active_session, timeout_seconds=timeout_seconds)
            for item in plan.items
        ]
    finally:
        if owns_session:
            active_session.close()


def download_file(
    item: DownloadItem,
    *,
    session: requests.Session,
    timeout_seconds: float = 60.0,
    chunk_size: int = 1024 * 1024,
) -> DownloadResult:
    """Download one URL safely, continuing an existing partial file when supported."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    destination = item.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        expected_size = _content_length(session, item.url, timeout_seconds)
        if expected_size is None or destination.stat().st_size == expected_size:
            return DownloadResult(item, 0, True)
        destination.unlink()
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = session.get(item.url, headers=headers, stream=True, timeout=timeout_seconds)
    try:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        if offset > 0 and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
        expected_size = _expected_response_size(response, offset)
    finally:
        response.close()
    actual_size = partial.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise OSError(
            f"incomplete download for {item.url}: expected {expected_size} bytes, got {actual_size}"
        )
    os.replace(partial, destination)
    return DownloadResult(item, actual_size - offset, False)


def extract_replica_parts(
    plan: DownloadPlan,
    destination: str | Path,
    *,
    remove_parts: bool = False,
) -> None:
    """Stream-extract Replica parts without materializing a second merged tarball."""

    if plan.dataset != "replica":
        raise ValueError("extract_replica_parts requires a Replica plan")
    part_paths = [item.destination for item in plan.items]
    missing = [path for path in part_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Replica archive parts are missing: {missing[:3]}")
    output_root = Path(destination)
    output_root.mkdir(parents=True, exist_ok=True)
    with (
        _ConcatenatedReader(part_paths) as stream,
        tarfile.open(fileobj=stream, mode="r|gz") as archive,
    ):
        for member in archive:
            target = safe_member_destination(output_root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise OSError(f"cannot read archive member {member.name}")
                with source, target.open("wb") as destination_handle:
                    shutil.copyfileobj(source, destination_handle, length=1024 * 1024)
            else:
                raise ValueError(f"refusing non-file Replica archive member: {member.name}")
    if remove_parts:
        for part in part_paths:
            part.unlink()


def download_hypersim_members(
    scene: str,
    destination: str | Path,
    *,
    camera: str = "cam_00",
    frame_ids: set[int] | None = None,
    include_position: bool = False,
    overwrite: bool = False,
    session: requests.Session | None = None,
    timeout_seconds: float = 60.0,
) -> list[Path]:
    """Extract a practical RGB/depth/normal subset directly from an official scene ZIP.

    ZIP central-directory access uses HTTP byte ranges, so the full 1-20 GB scene archive is not
    written locally. Camera trajectories are always retained; image modalities can be restricted
    to a set of frame ids for smoke and pilot experiments.
    """

    if not re.fullmatch(r"cam_\d{2}", camera):
        raise ValueError("camera must match cam_DD")
    root = Path(destination)
    active_session = session or requests.Session()
    owns_session = session is None
    try:
        with HTTPRangeReader(hypersim_scene_url(scene), active_session, timeout_seconds) as reader:
            with zipfile.ZipFile(reader) as archive:
                selected: list[tuple[zipfile.ZipInfo, str]] = []
                for info in archive.infolist():
                    relative_name = hypersim_member_relative_name(scene, info.filename)
                    if _select_hypersim_member(
                        relative_name,
                        camera=camera,
                        frame_ids=frame_ids,
                        include_position=include_position,
                    ):
                        selected.append((info, relative_name))
                written: list[Path] = []
                for info, relative_name in selected:
                    target = safe_member_destination(root / scene, relative_name)
                    if target.exists() and not overwrite:
                        written.append(target)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + ".part")
                    with archive.open(info) as source, partial.open("wb") as destination_handle:
                        shutil.copyfileobj(source, destination_handle, length=1024 * 1024)
                    os.replace(partial, target)
                    written.append(target)
                return written
    finally:
        if owns_session:
            active_session.close()


class HTTPRangeReader(io.RawIOBase):
    """Seekable read-only stream backed by HTTP Range requests for `zipfile.ZipFile`."""

    def __init__(self, url: str, session: requests.Session, timeout_seconds: float) -> None:
        super().__init__()
        self.url = url
        self.session = session
        self.timeout_seconds = timeout_seconds
        response = self.session.head(url, allow_redirects=True, timeout=timeout_seconds)
        try:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length is None:
                raise OSError(f"server did not provide content length for {url}")
            self.size = int(content_length)
        finally:
            response.close()
        self.offset = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.offset

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_offset = offset
        elif whence == io.SEEK_CUR:
            new_offset = self.offset + offset
        elif whence == io.SEEK_END:
            new_offset = self.size + offset
        else:
            raise ValueError(f"unsupported seek mode {whence}")
        self.offset = min(max(new_offset, 0), self.size)
        return self.offset

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self.offset >= self.size:
            return b""
        if size < 0:
            size = self.size - self.offset
        end = min(self.offset + size, self.size) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.offset}-{end}"},
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
            if response.status_code != 206:
                raise OSError(f"server ignored Range request for {self.url}")
            data = response.content
        finally:
            response.close()
        if len(data) > size:
            data = data[:size]
        self.offset += len(data)
        return data


class _ConcatenatedReader(io.RawIOBase):
    """Read several archive chunks as one file without a temporary merged copy."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self._index = 0
        self._handle: BinaryIO | None = None

    def __enter__(self) -> _ConcatenatedReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while self._index < len(self.paths) and (remaining != 0):
            if self._handle is None:
                self._handle = self.paths[self._index].open("rb")
            chunk = self._handle.read(remaining)
            if chunk:
                chunks.append(chunk)
                if remaining > 0:
                    remaining -= len(chunk)
            if remaining == 0:
                break
            if not chunk:
                self._handle.close()
                self._handle = None
                self._index += 1
        return b"".join(chunks)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def hypersim_member_relative_name(scene: str, member_name: str) -> str:
    """Remove the official scene-root prefix while rejecting cross-scene members."""

    if not _SCENE_NAME.fullmatch(scene):
        raise ValueError("scene must match ai_DDD_DDD")
    normalized = member_name.replace("\\", "/")
    prefix = f"{scene}/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    first_part = normalized.split("/", 1)[0]
    if _SCENE_NAME.fullmatch(first_part):
        raise ValueError(f"archive member belongs to a different scene: {member_name!r}")
    return normalized


def _select_hypersim_member(
    name: str, *, camera: str, frame_ids: set[int] | None, include_position: bool
) -> bool:
    if name in {"_detail/metadata_scene.csv", "_detail/metadata_cameras.csv"}:
        return True
    if name.startswith(f"_detail/{camera}/") and name.endswith(".hdf5"):
        return True
    preview_prefix = f"images/scene_{camera}_final_preview/"
    geometry_prefix = f"images/scene_{camera}_geometry_hdf5/"
    if not (name.startswith(preview_prefix) or name.startswith(geometry_prefix)):
        return False
    frame_match = re.search(r"frame\.(\d{4})\.", name)
    if frame_match is None:
        return False
    if frame_ids is not None and int(frame_match.group(1)) not in frame_ids:
        return False
    return (
        name.endswith(".color.jpg")
        or name.endswith(".depth_meters.hdf5")
        or name.endswith(".normal_world.hdf5")
        or (include_position and name.endswith(".position.hdf5"))
    )


def _content_length(session: requests.Session, url: str, timeout_seconds: float) -> int | None:
    response = session.head(url, allow_redirects=True, timeout=timeout_seconds)
    try:
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        return None if value is None else int(value)
    finally:
        response.close()


def _expected_response_size(response: requests.Response, offset: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range:
        match = re.fullmatch(r"bytes \d+-\d+/(\d+)", content_range)
        if match:
            return int(match.group(1))
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    return offset + int(content_length)
