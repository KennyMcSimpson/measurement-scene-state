"""Pinned, resumable access to the gated DL3DV-140 benchmark.

The official repository ships a helper that downloads a pickle-backed file list and
uses a mutable Hugging Face cache.  This module deliberately uses the Hub tree API
and direct ``resolve`` requests instead.  The resulting manifest is bound to one
immutable revision, and every downloaded object is checked against its Hub LFS
SHA256 or Git blob SHA1 identity.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

REPO_ID = "DL3DV/DL3DV-Benchmark"
REPO_TYPE = "dataset"
REVISION = "9684e8382278c5e18173c1e72bd246daf287453"
METADATA_PATH = "benchmark-meta.csv"
EXPECTED_SCENE_COUNT = 140
DEFAULT_REPRESENTATION = "nerfstudio"
DEFAULT_MIN_FREE_BYTES = 10 * 1024**3
DEFAULT_CHUNK_SIZE = 4 * 1024**2
DEFAULT_MAX_ATTEMPTS = 5

_SCENE_RE = re.compile(r"^[0-9a-f]{64}$")
_FRAME_RE = re.compile(r"(?:^|/)frame_(\d+)(?:\.[^/]*)?$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class Dl3dvError(RuntimeError):
    """Base error for the pinned DL3DV access layer."""


class IntegrityError(Dl3dvError):
    """A local or remote file did not match its immutable Hub metadata."""


class DownloadBusy(Dl3dvError):
    """Another process owns the destination lock."""


@dataclass(frozen=True)
class SceneMeta:
    """One row from the official 140-scene benchmark metadata CSV."""

    index: int
    hash: str
    fields: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "hash": self.hash, **self.fields}


@dataclass(frozen=True)
class RemoteFile:
    """Hub tree identity for one file, excluding untrusted cache entries."""

    path: str
    size: int
    git_blob_id: str | None
    lfs_sha256: str | None
    lfs_size: int | None
    scene_hash: str | None

    @classmethod
    def from_hf(cls, entry: Any) -> RemoteFile:
        path = str(getattr(entry, "path", getattr(entry, "rfilename", "")))
        if not path or path.replace("\\", "/").startswith(".cache/") or path == ".cache":
            raise ValueError(f"cache or empty path is not an inventory file: {path!r}")
        path = path.replace("\\", "/")
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"unsafe Hub path: {path!r}")
        raw_size = getattr(entry, "size", None)
        if raw_size is None:
            raise ValueError(f"Hub tree entry has no size: {path}")
        size = int(raw_size)
        if size < 0:
            raise ValueError(f"Hub tree entry has negative size: {path}")
        lfs = getattr(entry, "lfs", None)
        lfs_sha256 = _lfs_value(lfs, "sha256")
        lfs_size_raw = _lfs_value(lfs, "size")
        lfs_size = None if lfs_size_raw is None else int(lfs_size_raw)
        if lfs_sha256 is not None and not _HEX64_RE.fullmatch(lfs_sha256):
            raise ValueError(f"invalid LFS SHA256 for {path}")
        if lfs_size is not None and lfs_size != size:
            raise ValueError(f"LFS size differs from tree size for {path}")
        blob = getattr(entry, "blob_id", None)
        git_blob_id = None if blob is None else str(blob)
        if git_blob_id is not None and not _HEX40_RE.fullmatch(git_blob_id):
            raise ValueError(f"invalid Git blob id for {path}")
        first = path.split("/", 1)[0]
        scene_hash = first if _SCENE_RE.fullmatch(first) else None
        return cls(path, size, git_blob_id, lfs_sha256, lfs_size, scene_hash)

    @property
    def relative_path(self) -> str:
        if self.scene_hash is None:
            return self.path
        return self.path.split("/", 1)[1]

    @property
    def kind(self) -> str:
        relative = self.relative_path
        if "/images_4/" in f"/{relative}/":
            return "images_4"
        if relative.endswith("/transforms.json"):
            return "transforms"
        if relative.endswith("/sparse/0/cameras.bin"):
            return "cameras"
        if relative.endswith("/sparse/0/images.bin"):
            return "colmap_images"
        if relative.endswith("/sparse/0/points3D.bin"):
            return "colmap_points"
        return "other"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "git_blob_id": self.git_blob_id,
            "lfs_sha256": self.lfs_sha256,
            "lfs_size": self.lfs_size,
            "scene_hash": self.scene_hash,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Inventory:
    """Complete non-cache tree inventory for one pinned Hub revision."""

    metadata_bytes: bytes
    metadata_file: RemoteFile
    scenes: tuple[SceneMeta, ...]
    files: tuple[RemoteFile, ...]
    excluded_scene_hashes: tuple[str, ...] = ()

    @property
    def scene_hashes(self) -> tuple[str, ...]:
        return tuple(scene.hash for scene in self.scenes)

    def summary(self, representation: str = DEFAULT_REPRESENTATION) -> dict[str, Any]:
        selected = [file for file in self.files if file.scene_hash in self.scene_hashes]
        canonical = [
            file
            for file in selected
            if file.relative_path.startswith(f"{representation}/")
        ]
        images_all = [file for file in selected if file.kind == "images_4"]
        images_canonical = [file for file in canonical if file.kind == "images_4"]
        transforms = [file for file in canonical if file.kind == "transforms"]
        cameras = [file for file in canonical if file.kind == "cameras"]
        return {
            "scene_count": len(self.scenes),
            "excluded_root_scene_folder_count": len(self.excluded_scene_hashes),
            "excluded_root_scene_hashes": list(self.excluded_scene_hashes),
            "non_cache_file_count": len(selected),
            "non_cache_bytes": sum(file.size for file in selected),
            "images_4_all_representations": _size_summary(images_all),
            "images_4_canonical": _size_summary(images_canonical),
            "transforms_json_canonical": _size_summary(transforms),
            "cameras_bin_canonical": _size_summary(cameras),
            "camera_metadata_canonical": _size_summary(transforms + cameras),
        }

    def manifest(self, representation: str = DEFAULT_REPRESENTATION) -> dict[str, Any]:
        metadata_sha256 = hashlib.sha256(self.metadata_bytes).hexdigest()
        scene_rows = []
        for scene in self.scenes:
            files = [file for file in self.files if file.scene_hash == scene.hash]
            scene_rows.append(
                {
                    **scene.to_dict(),
                    "non_cache_file_count": len(files),
                    "non_cache_bytes": sum(file.size for file in files),
                    "images_4_canonical": _size_summary(
                        [
                            file
                            for file in files
                            if file.kind == "images_4"
                            and file.relative_path.startswith(f"{representation}/")
                        ]
                    ),
                    "camera_metadata_canonical": _size_summary(
                        [
                            file
                            for file in files
                            if file.kind in {"transforms", "cameras"}
                            and file.relative_path.startswith(f"{representation}/")
                        ]
                    ),
                }
            )
        return {
            "schema": "mcss.dl3dv.inventory.v1",
            "repo_id": REPO_ID,
            "repo_type": REPO_TYPE,
            "revision": REVISION,
            "metadata_path": METADATA_PATH,
            "metadata_size": self.metadata_file.size,
            "metadata_git_blob_id": self.metadata_file.git_blob_id,
            "metadata_sha256": metadata_sha256,
            "representation": representation,
            "scene_count": len(self.scenes),
            "excluded_root_scene_folder_count": len(self.excluded_scene_hashes),
            "excluded_root_scene_hashes": list(self.excluded_scene_hashes),
            "scene_hashes": list(self.scene_hashes),
            "summary": self.summary(representation),
            "scenes": scene_rows,
        }


@dataclass(frozen=True)
class SelectedFile:
    remote: RemoteFile
    destination: Path


def _lfs_value(lfs: Any, name: str) -> Any:
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get(name)
    return getattr(lfs, name, None)


def _size_summary(files: Sequence[RemoteFile]) -> dict[str, int]:
    return {"files": len(files), "bytes": sum(file.size for file in files)}


def parse_benchmark_metadata(
    payload: bytes, *, expected_count: int = EXPECTED_SCENE_COUNT
) -> tuple[SceneMeta, ...]:
    """Parse and validate the official benchmark CSV without pandas or pickle."""

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("benchmark-meta.csv is not UTF-8") from error
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows:
        raise ValueError("benchmark-meta.csv is empty")
    if "hash" not in rows[0]:
        raise ValueError("benchmark-meta.csv has no hash column")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected exactly {expected_count} benchmark scenes, got {len(rows)}")
    scenes: list[SceneMeta] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        scene_hash = str(row.get("hash", "")).strip()
        if not _SCENE_RE.fullmatch(scene_hash):
            raise ValueError(f"invalid scene hash at CSV row {index + 2}: {scene_hash!r}")
        if scene_hash in seen:
            raise ValueError(f"duplicate scene hash: {scene_hash}")
        seen.add(scene_hash)
        fields = {key: str(value or "") for key, value in row.items() if key != "hash"}
        scenes.append(SceneMeta(index=index, hash=scene_hash, fields=fields))
    return tuple(scenes)


def _hub_api(token: str | None = None) -> tuple[Any, str]:
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError("huggingface_hub is required for DL3DV inventory/downloads") from error
    effective = token if token is not None else get_token()
    if not effective:
        raise RuntimeError("no Hugging Face token is available; run hf auth login first")
    return HfApi(token=effective), effective


def hf_resolve_url(remote: RemoteFile | str) -> str:
    path = remote.path if isinstance(remote, RemoteFile) else remote
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"unsafe Hub path: {path!r}")
    return (
        f"https://huggingface.co/{REPO_TYPE}s/{REPO_ID}/resolve/"
        f"{REVISION}/{quote(path, safe='/')}?download=true"
    )


def git_blob_sha1(path: str | Path) -> str:
    """Return the SHA1 used by Git for a regular (non-LFS) blob."""

    file_path = Path(path)
    size = file_path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DEFAULT_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git = hashlib.sha1()
    size = path.stat().st_size
    git.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DEFAULT_CHUNK_SIZE), b""):
            sha256.update(chunk)
            git.update(chunk)
    return sha256.hexdigest(), git.hexdigest()


def verify_local_file(path: str | Path, remote: RemoteFile) -> dict[str, Any]:
    """Verify size and the strongest Hub identity available for ``path``."""

    file_path = Path(path)
    if not file_path.is_file():
        raise IntegrityError(f"missing local file: {file_path}")
    actual_size = file_path.stat().st_size
    if actual_size != remote.size:
        raise IntegrityError(
            f"size mismatch for {file_path}: expected {remote.size}, got {actual_size}"
        )
    sha256, git_sha1 = _hash_file(file_path)
    if remote.lfs_sha256 is not None and sha256 != remote.lfs_sha256:
        raise IntegrityError(f"LFS SHA256 mismatch for {file_path}")
    if (
        remote.lfs_sha256 is None
        and remote.git_blob_id is not None
        and git_sha1 != remote.git_blob_id
    ):
        raise IntegrityError(f"Git blob SHA1 mismatch for {file_path}")
    return {"bytes": actual_size, "sha256": sha256, "git_blob_sha1": git_sha1}


def _api_paths_info(api: Any, paths: Sequence[str]) -> list[Any]:
    return list(
        api.get_paths_info(
            REPO_ID,
            list(paths),
            expand=False,
            revision=REVISION,
            repo_type=REPO_TYPE,
        )
    )


def _scene_hashes_from_tree(api: Any) -> tuple[str, ...]:
    rows = list(
        api.list_repo_tree(
            REPO_ID,
            path_in_repo="",
            recursive=False,
            expand=False,
            revision=REVISION,
            repo_type=REPO_TYPE,
        )
    )
    hashes = sorted(
        str(getattr(row, "path", ""))
        for row in rows
        if type(row).__name__ == "RepoFolder" and _SCENE_RE.fullmatch(str(getattr(row, "path", "")))
    )
    return tuple(hashes)


def _scene_tree(api: Any, scene_hash: str) -> list[RemoteFile]:
    rows = api.list_repo_tree(
        REPO_ID,
        path_in_repo=scene_hash,
        recursive=True,
        expand=False,
        revision=REVISION,
        repo_type=REPO_TYPE,
    )
    result = []
    for row in rows:
        if type(row).__name__ != "RepoFile":
            continue
        path = str(getattr(row, "path", ""))
        if path.startswith(".cache/") or path == ".cache":
            continue
        remote = RemoteFile.from_hf(row)
        if remote.scene_hash != scene_hash:
            raise ValueError(f"tree entry crossed scene boundary: {remote.path}")
        result.append(remote)
    return result


def _fetch_bytes(
    remote: RemoteFile,
    *,
    token: str,
    session: requests.Session,
    timeout: tuple[float, float] = (20.0, 120.0),
) -> bytes:
    headers = {"Authorization": f"Bearer {token}"}
    response = session.get(hf_resolve_url(remote), headers=headers, timeout=timeout)
    try:
        response.raise_for_status()
        payload = response.content
    finally:
        response.close()
    if len(payload) != remote.size:
        raise IntegrityError(
            f"remote size mismatch for {remote.path}: expected {remote.size}, got {len(payload)}"
        )
    sha256 = hashlib.sha256(payload).hexdigest()
    if remote.lfs_sha256 is not None and sha256 != remote.lfs_sha256:
        raise IntegrityError(f"remote LFS SHA256 mismatch for {remote.path}")
    if remote.lfs_sha256 is None and remote.git_blob_id is not None:
        prefix = f"blob {len(payload)}\0".encode("ascii")
        if hashlib.sha1(prefix + payload).hexdigest() != remote.git_blob_id:
            raise IntegrityError(f"remote Git blob SHA1 mismatch for {remote.path}")
    return payload


def build_inventory(
    *,
    token: str | None = None,
    api: Any | None = None,
    workers: int = 6,
) -> Inventory:
    """Fetch the pinned metadata and complete non-cache tree inventory.

    The tree calls use ``expand=False`` because file size, Git blob id and LFS
    SHA256 are already present in the normal tree response.  No repository cache
    contents are requested or deserialized.
    """

    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    if api is None:
        api, effective_token = _hub_api(token)
    else:
        effective_token = token
        if effective_token is None:
            try:
                _, effective_token = _hub_api(None)
            except RuntimeError:
                effective_token = ""
    metadata_entries = _api_paths_info(api, [METADATA_PATH])
    if len(metadata_entries) != 1:
        raise ValueError("pinned repository did not return benchmark-meta.csv")
    metadata_file = RemoteFile.from_hf(metadata_entries[0])
    if metadata_file.path != METADATA_PATH or effective_token == "":
        raise ValueError("metadata file or token is unavailable")
    with requests.Session() as session:
        metadata_bytes = _fetch_bytes(metadata_file, token=effective_token, session=session)
    scenes = parse_benchmark_metadata(metadata_bytes)
    root_hashes = _scene_hashes_from_tree(api)
    expected_hashes = set(scene.hash for scene in scenes)
    if not expected_hashes.issubset(set(root_hashes)):
        missing = sorted(expected_hashes - set(root_hashes))
        raise ValueError(
            f"scene folder/metadata mismatch; missing={missing[:3]}"
        )
    folder_hashes = tuple(sorted(expected_hashes))
    excluded = tuple(sorted(set(root_hashes) - expected_hashes))
    files: list[RemoteFile] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(folder_hashes))) as pool:
        futures = {
            pool.submit(_scene_tree, api, scene_hash): scene_hash for scene_hash in folder_hashes
        }
        for future in as_completed(futures):
            files.extend(future.result())
    files.sort(key=lambda file: file.path)
    if any(file.scene_hash is None for file in files):
        raise ValueError("inventory contains a file outside a scene folder")
    return Inventory(metadata_bytes, metadata_file, scenes, tuple(files), excluded)


def _frame_number(path: str) -> int | None:
    match = _FRAME_RE.search(path)
    return None if match is None else int(match.group(1))


def select_files(
    inventory: Inventory,
    destination_root: str | Path,
    *,
    scenes: Iterable[str] | None = None,
    representation: str = DEFAULT_REPRESENTATION,
    include_images4: bool = False,
    frame_ids: Iterable[int] | None = None,
    frame_ids_by_scene: dict[str, Iterable[int]] | None = None,
    include_cameras: bool = False,
) -> tuple[SelectedFile, ...]:
    """Select canonical metadata and optional level-4 frames without duplicates."""

    if not representation or "/" in representation or "\\" in representation:
        raise ValueError("representation must be a single path component")
    selected_scenes = set(inventory.scene_hashes if scenes is None else scenes)
    unknown = selected_scenes - set(inventory.scene_hashes)
    if unknown:
        raise ValueError(f"unknown DL3DV scene hash: {sorted(unknown)[0]}")
    if frame_ids is not None and frame_ids_by_scene is not None:
        raise ValueError("pass either frame_ids or frame_ids_by_scene, not both")
    frames = None if frame_ids is None else {int(frame) for frame in frame_ids}
    per_scene = (
        None
        if frame_ids_by_scene is None
        else {
            scene: {int(frame) for frame in values}
            for scene, values in frame_ids_by_scene.items()
        }
    )
    if per_scene is not None and not set(per_scene).issubset(selected_scenes):
        raise ValueError("frame_ids_by_scene contains an unknown scene hash")
    all_frame_values = frames if frames is not None else {
        frame for values in (per_scene or {}).values() for frame in values
    }
    if any(frame < 0 for frame in all_frame_values):
        raise ValueError("frame ids must be non-negative")
    root = Path(destination_root)
    output: list[SelectedFile] = []
    seen: set[str] = set()
    for remote in inventory.files:
        if remote.scene_hash not in selected_scenes:
            continue
        relative = remote.relative_path
        if not relative.startswith(f"{representation}/"):
            continue
        include = False
        if remote.kind == "transforms":
            include = True
        elif include_cameras and remote.kind == "cameras":
            include = True
        elif include_images4 and remote.kind == "images_4":
            frame = _frame_number(relative)
            scene_frames = (
                None if per_scene is None else per_scene.get(remote.scene_hash, set())
            )
            include = (
                (frames is None and per_scene is None)
                or (frame is not None and frames is not None and frame in frames)
                or (frame is not None and scene_frames is not None and frame in scene_frames)
            )
        if not include or remote.path in seen:
            continue
        seen.add(remote.path)
        destination = root / str(remote.scene_hash) / relative
        _ensure_descendant(root, destination)
        output.append(SelectedFile(remote, destination))
    if include_images4 and not any(item.remote.kind == "images_4" for item in output):
        raise ValueError("no selected images_4 files matched the scene/frame filter")
    return tuple(output)


def protocol_frame_selection(
    protocol_path: str | Path,
    inventory: Inventory,
    *,
    data_root: str | Path,
    metadata_root: str | Path | None = None,
    representation: str = DEFAULT_REPRESENTATION,
    input_keys: Sequence[str] = (
        "fold_8_kmeans_16_input",
        "fold_8_kmeans_32_input",
    ),
    target_every: int = 8,
) -> dict[str, set[int]]:
    """Map protocol frame positions to the real Hub image frame numbers.

    Protocol values index the ordered ``frames`` list in each scene's local
    ``transforms.json``.  The converter keeps the basename from each
    ``file_path`` under ``images_4``; therefore the file number cannot be
    inferred from the protocol position.  ``data_root`` is the local DL3DV
    scene root, while ``metadata_root`` can point at a separate root containing
    the same scene metadata.  Every transforms file is checked against its
    pinned inventory identity before it is parsed.
    """

    if target_every < 1:
        raise ValueError("target_every must be positive")
    if not representation or "/" in representation or "\\" in representation:
        raise ValueError("representation must be a single path component")
    try:
        rows = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read protocol frame JSON: {protocol_path}") from error
    if not isinstance(rows, list):
        raise ValueError("protocol frame JSON must contain a list")
    inventory_scenes = set(inventory.scene_hashes)
    inventory_by_path: dict[str, list[RemoteFile]] = {}
    for remote in inventory.files:
        inventory_by_path.setdefault(remote.path, []).append(remote)
    metadata_base = Path(data_root if metadata_root is None else metadata_root)
    frame_ids_by_scene: dict[str, tuple[int, ...]] = {}

    def load_scene_frame_ids(scene: str) -> tuple[int, ...]:
        cached = frame_ids_by_scene.get(scene)
        if cached is not None:
            return cached
        transform_files = [
            file
            for file in inventory_by_path.get(f"{scene}/{representation}/transforms.json", ())
            if file.kind == "transforms"
        ]
        if len(transform_files) != 1:
            raise ValueError(
                f"inventory must contain exactly one transforms target for scene: {scene}"
            )
        transform_path = metadata_base / scene / representation / "transforms.json"
        try:
            verify_local_file(transform_path, transform_files[0])
            transforms = json.loads(transform_path.read_text(encoding="utf-8"))
        except IntegrityError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read transforms metadata for scene: {scene}") from error
        if not isinstance(transforms, dict):
            raise ValueError(f"transforms metadata must contain an object: {scene}")
        raw_frames = transforms.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError(f"transforms metadata has no non-empty frames list: {scene}")

        frame_ids: list[int] = []
        seen_paths: set[str] = set()
        for index, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, dict):
                raise ValueError(f"transforms frame {index} is not an object: {scene}")
            raw_path = raw_frame.get("file_path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"transforms frame {index} has no file_path: {scene}")
            normalized = raw_path.replace("\\", "/")
            candidate = Path(normalized)
            parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
            if candidate.is_absolute() or any(part == ".." for part in parts):
                raise ValueError(f"unsafe transforms frame file_path: {raw_path!r}")
            if len(parts) < 2:
                raise ValueError(
                    f"transforms frame file_path must point to an image filename: {scene}"
                )
            relative = f"{representation}/images_4/{parts[-1]}"
            if relative in seen_paths:
                raise ValueError(f"duplicate frame path in transforms metadata: {scene}/{relative}")
            seen_paths.add(relative)
            targets = [
                file
                for file in inventory_by_path.get(f"{scene}/{relative}", ())
                if file.kind == "images_4"
            ]
            if len(targets) != 1:
                raise ValueError(
                    f"missing inventory target for transforms frame: {scene}/{relative}"
                )
            frame_number = _frame_number(relative)
            if frame_number is None:
                raise ValueError(f"transforms frame path has no frame number: {scene}/{relative}")
            frame_ids.append(frame_number)
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError(f"duplicate frame number in transforms metadata: {scene}")
        result = tuple(frame_ids)
        frame_ids_by_scene[scene] = result
        return result

    selection: dict[str, set[int]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("scene_name"), str):
            raise ValueError("protocol frame row has no scene_name")
        scene = row["scene_name"]
        if scene not in inventory_scenes:
            raise ValueError(f"protocol frame row has unknown scene: {scene}")
        frame_ids = load_scene_frame_ids(scene)
        values: set[int] = set()
        for key in input_keys:
            raw_values = row.get(key)
            if not isinstance(raw_values, list):
                raise ValueError(f"protocol frame row has no list for {key}: {scene}")
            for value in raw_values:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"invalid zero-based frame index in {key}: {scene}")
                if value >= len(frame_ids):
                    raise ValueError(
                        f"zero-based frame index out of range in {key}: {scene} index={value}"
                    )
                values.add(frame_ids[value])
        values.update(frame_ids[index] for index in range(0, len(frame_ids), target_every))
        if scene in selection and selection[scene] != values:
            raise ValueError(f"duplicate protocol row for scene: {scene}")
        selection[scene] = values
    if set(selection) != inventory_scenes:
        missing = sorted(inventory_scenes - set(selection))
        extra = sorted(set(selection) - inventory_scenes)
        raise ValueError(
            f"protocol scene coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    return selection


def _ensure_descendant(root: Path, destination: Path) -> None:
    base = root.resolve()
    target = destination.resolve()
    try:
        target.relative_to(base)
    except ValueError as error:
        raise ValueError(f"destination escapes download root: {destination}") from error


def _free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _check_free_space(path: Path, required: int, reserve: int) -> None:
    if required < 0 or reserve < 0:
        raise ValueError("required and reserve bytes must be non-negative")
    free = _free_bytes(path)
    if free < required + reserve:
        raise OSError(
            f"insufficient free space below {path}: need {required + reserve}, have {free}"
        )


def atomic_write_json(path: str | Path, value: Any, *, attempts: int = 6) -> None:
    """Write status/manifests through a replace with short Windows retry backoff."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, destination)
            return
        except PermissionError as error:
            last_error = error
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1 * (attempt + 1))
    if last_error is not None:  # pragma: no cover
        raise last_error


def atomic_write_bytes(path: str | Path, payload: bytes, *, attempts: int = 6) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    for attempt in range(attempts):
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1 * (attempt + 1))


class ExclusiveLock(AbstractContextManager["ExclusiveLock"]):
    """Cross-process non-blocking lock held for one inventory/download run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: Any | None = None

    def __enter__(self) -> ExclusiveLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - CI for this project is Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, PermissionError) as error:
            handle.close()
            raise DownloadBusy(f"DL3DV destination is already locked: {self.path}") from error
        self._handle = handle
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is None:
            return
        try:
            if os.name != "nt":  # pragma: no cover - CI for this project is Windows
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _download_one(
    selected: SelectedFile,
    *,
    token: str,
    session: requests.Session,
    reserve: int,
    chunk_size: int,
    max_attempts: int,
) -> dict[str, Any]:
    remote, destination = selected.remote, selected.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_root = destination.parent.parent.parent if remote.scene_hash else destination.parent
    _ensure_descendant(destination_root, destination)
    if destination.exists():
        verification = verify_local_file(destination, remote)
        return {"status": "reused", **verification}
    partial = destination.with_name(destination.name + ".part")
    url = hf_resolve_url(remote)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > remote.size:
                raise IntegrityError(f"partial file exceeds expected size: {partial}")
            if offset == remote.size:
                verification = verify_local_file(partial, remote)
                os.replace(partial, destination)
                return {"status": "downloaded", "attempts": attempt, **verification}
            _check_free_space(destination.parent, remote.size - offset, reserve)
            headers = {"Authorization": f"Bearer {token}"}
            if offset:
                headers["Range"] = f"bytes={offset}-{remote.size - 1}"
            response = session.get(url, headers=headers, stream=True, timeout=(20.0, 120.0))
            try:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise OSError("Hub did not honor the resume Range request")
                if offset:
                    content_range = response.headers.get("Content-Range", "")
                    expected_range = f"bytes {offset}-{remote.size - 1}/{remote.size}"
                    if content_range != expected_range:
                        raise OSError(f"unexpected resume Content-Range: {content_range!r}")
                mode = "ab" if offset or partial.exists() else "xb"
                with partial.open(mode) as handle:
                    written = offset
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        if written + len(chunk) > remote.size:
                            raise IntegrityError(
                                f"response exceeded expected size for {remote.path}"
                            )
                        if _free_bytes(destination.parent) < reserve:
                            raise OSError(f"free-space reserve reached below {destination.parent}")
                        handle.write(chunk)
                        written += len(chunk)
            finally:
                response.close()
            verification = verify_local_file(partial, remote)
            os.replace(partial, destination)
            return {"status": "downloaded", "attempts": attempt, **verification}
        except (OSError, IntegrityError, requests.RequestException) as error:
            last_error = error
            if attempt < max_attempts:
                time.sleep(min(30.0, 0.5 * 2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def download_selected(
    selected: Sequence[SelectedFile],
    *,
    token: str | None = None,
    status_path: str | Path | None = None,
    lock_path: str | Path | None = None,
    reserve_bytes: int = DEFAULT_MIN_FREE_BYTES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    workers: int = 8,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download selected files with bounded concurrency under one exclusive lock.

    Worker threads only transfer and verify one file at a time.  The caller
    thread owns the status snapshot, so each update is deterministic and
    atomically replaced.  Network retries happen per file, and every partial
    file remains available for a later resume.
    """

    if reserve_bytes < 0 or chunk_size < 1 or max_attempts < 1:
        raise ValueError("reserve_bytes, chunk_size and max_attempts are invalid")
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    if not selected:
        return {"schema": "mcss.dl3dv.download.v1", "status": "complete", "entries": []}
    root = Path(lock_path).parent if lock_path is not None else selected[0].destination.parents[2]
    lock = Path(lock_path) if lock_path is not None else root / ".dl3dv.lock"
    status = Path(status_path) if status_path is not None else root / "status.json"
    root.mkdir(parents=True, exist_ok=True)
    space_path = selected[0].destination.parent
    space_path.mkdir(parents=True, exist_ok=True)
    total_pending = sum(
        max(
            0,
            item.remote.size
            - (item.destination.stat().st_size if item.destination.exists() else 0),
        )
        for item in selected
    )
    _check_free_space(space_path, total_pending, reserve_bytes)
    if token is None:
        _, token = _hub_api(None)
    entries = [
        {
            "repo_path": item.remote.path,
            "destination": str(item.destination),
            "expected_bytes": item.remote.size,
            "status": "queued",
        }
        for item in selected
    ]
    started = time.time()

    def snapshot(state: str, error: str | None = None) -> dict[str, Any]:
        payload = {
            "schema": "mcss.dl3dv.download.v1",
            "status": state,
            "repo_id": REPO_ID,
            "repo_type": REPO_TYPE,
            "revision": REVISION,
            "started_unix": started,
            "updated_unix": time.time(),
            "reserve_bytes": reserve_bytes,
            "entries": entries,
        }
        if error:
            payload["error"] = error
        return payload

    worker_local = threading.local()
    worker_sessions: list[requests.Session] = []
    worker_sessions_lock = threading.Lock()

    def download_one(index: int, item: SelectedFile) -> tuple[int, dict[str, Any]]:
        if session is not None:
            active_session = session
        else:
            active_session = getattr(worker_local, "session", None)
            if active_session is None:
                active_session = requests.Session()
                worker_local.session = active_session
                with worker_sessions_lock:
                    worker_sessions.append(active_session)
        result = _download_one(
            item,
            token=token,
            session=active_session,
            reserve=reserve_bytes,
            chunk_size=chunk_size,
            max_attempts=max_attempts,
        )
        return index, result

    try:
        with ExclusiveLock(lock):
            if status.exists():
                # Existing status is informational; local files are always verified again.
                try:
                    json.loads(status.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(f"existing status is not valid JSON: {status}") from error
            if status_path is not None:
                atomic_write_json(status, snapshot("running"))
            first_error: Exception | None = None
            with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as pool:
                futures = {
                    pool.submit(download_one, index, item): index
                    for index, item in enumerate(selected)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        _, result = future.result()
                    except Exception as error:
                        entries[index].update({"status": "failed", "error": str(error)})
                        if first_error is None:
                            first_error = error
                    else:
                        entries[index].update(result)
                    if status_path is not None:
                        state = "failed" if first_error is not None else "running"
                        atomic_write_json(
                            status,
                            snapshot(state, None if first_error is None else str(first_error)),
                        )
            if first_error is not None:
                if status_path is not None:
                    atomic_write_json(status, snapshot("failed", str(first_error)))
                raise first_error
            result = snapshot("complete")
            if status_path is not None:
                atomic_write_json(status, result)
            return result
    finally:
        for worker_session in worker_sessions:
            worker_session.close()


def write_inventory_artifacts(
    inventory: Inventory,
    output_dir: str | Path,
    *,
    representation: str = DEFAULT_REPRESENTATION,
) -> dict[str, Path]:
    """Write the deterministic manifest and JSONL tree inventory."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / METADATA_PATH
    atomic_write_bytes(metadata_path, inventory.metadata_bytes)
    tree_path = root / "tree_inventory.jsonl"
    temporary = tree_path.with_name(f".{tree_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for remote in inventory.files:
            handle.write(json.dumps(remote.to_dict(), sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(temporary, tree_path)
    manifest_path = root / "manifest.json"
    atomic_write_json(manifest_path, inventory.manifest(representation))
    return {"metadata": metadata_path, "tree": tree_path, "manifest": manifest_path}


def load_inventory_artifacts(output_dir: str | Path) -> Inventory:
    """Load a previously written manifest/tree without contacting Hugging Face."""

    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("revision") != REVISION or manifest.get("repo_id") != REPO_ID:
        raise ValueError("inventory manifest is not bound to the pinned DL3DV revision")
    metadata_bytes = (root / METADATA_PATH).read_bytes()
    metadata_file = RemoteFile(
        METADATA_PATH,
        int(manifest["metadata_size"]),
        manifest.get("metadata_git_blob_id"),
        None,
        None,
        None,
    )
    if hashlib.sha256(metadata_bytes).hexdigest() != manifest.get("metadata_sha256"):
        raise IntegrityError("benchmark-meta.csv does not match manifest SHA256")
    scenes = parse_benchmark_metadata(metadata_bytes)
    files = []
    with (root / "tree_inventory.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                files.append(RemoteFile(
                    row["path"],
                    int(row["size"]),
                    row.get("git_blob_id"),
                    row.get("lfs_sha256"),
                    row.get("lfs_size"),
                    row.get("scene_hash"),
                ))
    excluded = tuple(manifest.get("excluded_root_scene_hashes", ()))
    return Inventory(metadata_bytes, metadata_file, scenes, tuple(files), excluded)


__all__ = [
    "DEFAULT_MIN_FREE_BYTES",
    "DEFAULT_REPRESENTATION",
    "EXPECTED_SCENE_COUNT",
    "Inventory",
    "METADATA_PATH",
    "RemoteFile",
    "REVISION",
    "REPO_ID",
    "REPO_TYPE",
    "SceneMeta",
    "SelectedFile",
    "Dl3dvError",
    "DownloadBusy",
    "ExclusiveLock",
    "IntegrityError",
    "atomic_write_bytes",
    "atomic_write_json",
    "build_inventory",
    "download_selected",
    "git_blob_sha1",
    "hf_resolve_url",
    "load_inventory_artifacts",
    "parse_benchmark_metadata",
    "protocol_frame_selection",
    "select_files",
    "verify_local_file",
    "write_inventory_artifacts",
]
