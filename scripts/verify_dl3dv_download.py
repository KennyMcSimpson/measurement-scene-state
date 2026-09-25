"""Independently verify the approved DL3DV download selection.

The verifier rebuilds the selection from the pinned inventory and protocol,
checks every local file against its Hub Git/LFS identity, and records local
SHA-256 values as ``local_only`` evidence.  A local SHA-256 is never treated
as an official source checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mcss.data.dl3dv_download import (
    DEFAULT_REPRESENTATION,
    METADATA_PATH,
    REPO_ID,
    REPO_TYPE,
    REVISION,
    atomic_write_json,
    load_inventory_artifacts,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "outputs/dl3dv_download_20260920"
DEFAULT_DATA_ROOT = PROJECT / "data/dl3dv_benchmark"
DEFAULT_PROTOCOL = (
    PROJECT
    / "outputs/benchmark_protocol_20260920/sources/Long-LRM/data/"
    / "dl3dv_fold_8_kmeans_input_idx.json"
)
EXPECTED_PROTOCOL_SHA256 = "4b052aef605fda183683175de66250aaaf337e50003131dbf0113ea781a3145a"
EXPECTED_FILE_COUNT = 12_777
EXPECTED_TOTAL_BYTES = 11_313_259_491
EXPECTED_RETAINED_EXTRA_FILES = 281
EXPECTED_RETAINED_EXTRA_BYTES = 246_096_151
EXPECTED_PHYSICAL_FILE_COUNT = 13_058
EXPECTED_PHYSICAL_BYTES = 11_559_355_642
PROTOCOL_INPUT_KEYS = (
    "fold_8_kmeans_16_input",
    "fold_8_kmeans_32_input",
)
PROTOCOL_NAMES = ("full16", "ar32")
FRAME_NAME_RE = re.compile(r"(?:^|/)frame_(\d+)(?:\.[^/]*)?$")
HASH_CHUNK_SIZE = 4 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT / "completion_verification.json"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--protocol-json", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-every", type=int, default=8)
    return parser


def _source_identity(remote: Any) -> dict[str, str | None]:
    if remote.lfs_sha256 is not None:
        return {"kind": "hf_lfs_sha256", "value": remote.lfs_sha256}
    if remote.git_blob_id is not None:
        return {"kind": "git_blob_sha1", "value": remote.git_blob_id}
    return {"kind": None, "value": None}


def _hash_local_file(path: Path) -> tuple[int, str, str]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1()
    git_sha1.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            sha256.update(chunk)
            git_sha1.update(chunk)
    return size, sha256.hexdigest(), git_sha1.hexdigest()


def _verify_remote(remote: Any, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing local file: {path}")
    actual_size, local_sha256, local_git_sha1 = _hash_local_file(path)
    if actual_size != remote.size:
        raise ValueError(
            f"size mismatch for {path}: expected {remote.size}, got {actual_size}"
        )
    source = _source_identity(remote)
    if source["value"] is None:
        raise ValueError(f"remote entry has no source identity: {remote.path}")
    if source["kind"] == "hf_lfs_sha256":
        source_hash_verified = local_sha256 == source["value"]
    else:
        source_hash_verified = local_git_sha1 == source["value"]
    if not source_hash_verified:
        raise ValueError(f"source identity mismatch for {remote.path}")
    return {
        "repo_path": remote.path,
        "relative_path": path.as_posix(),
        "bytes": actual_size,
        "source_hash_kind": source["kind"],
        "source_hash": source["value"],
        "source_hash_verified": True,
        "local_sha256": local_sha256,
        "local_sha256_role": "local_only",
    }


def _verify_selected(item: Any) -> dict[str, Any]:
    """Keep the small item-shaped helper for callers of the old verifier API."""

    return _verify_remote(item.remote, item.destination)


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"local path escapes data root: {path}") from error


def _destination(root: Path, remote: Any) -> Path:
    if remote.scene_hash is None:
        raise ValueError(f"scene file has no scene hash: {remote.path}")
    path = root / remote.scene_hash / remote.relative_path
    _relative_to_root(root, path)
    return path


def _frame_basename(raw_path: str, *, scene: str, index: int) -> str:
    normalized = raw_path.replace("\\", "/")
    candidate = Path(normalized)
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    if (
        not parts
        or candidate.is_absolute()
        or normalized.startswith("/")
        or any(part == ".." for part in parts)
        or ":" in parts[0]
        or len(parts) < 2
    ):
        raise ValueError(f"unsafe transforms frame file_path at index {index}: {scene}")
    basename = parts[-1]
    if FRAME_NAME_RE.fullmatch(basename) is None:
        raise ValueError(f"transforms frame path has no frame number: {scene}/{basename}")
    return basename


def _load_verified_frame_paths(
    scene: str,
    inventory: Any,
    data_root: Path,
    remote_by_path: dict[str, Any],
    verified_by_repo: dict[str, dict[str, Any]],
    *,
    representation: str = DEFAULT_REPRESENTATION,
) -> tuple[str, ...]:
    transform_path = f"{scene}/{representation}/transforms.json"
    transform = remote_by_path.get(transform_path)
    if transform is None or transform.kind != "transforms":
        raise ValueError(f"inventory has no canonical transforms target: {scene}")
    local_path = _destination(data_root, transform)
    transform_entry = _verify_remote(transform, local_path)
    transform_entry["relative_path"] = _relative_to_root(data_root, local_path)
    verified_by_repo[transform.path] = transform_entry
    try:
        transforms = json.loads(local_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read transforms metadata for scene: {scene}") from error
    if not isinstance(transforms, dict):
        raise ValueError(f"transforms metadata must contain an object: {scene}")
    raw_frames = transforms.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"transforms metadata has no non-empty frames list: {scene}")

    frame_paths: list[str] = []
    seen: set[str] = set()
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            raise ValueError(f"transforms frame {index} is not an object: {scene}")
        raw_path = raw_frame.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"transforms frame {index} has no file_path: {scene}")
        basename = _frame_basename(raw_path, scene=scene, index=index)
        relative = f"{representation}/images_4/{basename}"
        if relative in seen:
            raise ValueError(f"duplicate transforms frame path: {scene}/{relative}")
        seen.add(relative)
        remote_path = f"{scene}/{relative}"
        image = remote_by_path.get(remote_path)
        if image is None or image.kind != "images_4":
            raise ValueError(f"missing inventory target for transforms frame: {remote_path}")
        frame_paths.append(relative)
    return tuple(frame_paths)


def _build_required_paths(
    protocol_path: Path,
    inventory: Any,
    data_root: Path,
    verified_by_repo: dict[str, dict[str, Any]],
    *,
    target_every: int,
    representation: str = DEFAULT_REPRESENTATION,
) -> tuple[set[str], dict[str, Any]]:
    """Build the required set directly from verified transforms frame order."""

    if target_every < 1:
        raise ValueError("target_every must be positive")
    try:
        rows = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read protocol frame JSON: {protocol_path}") from error
    if not isinstance(rows, list):
        raise ValueError("protocol frame JSON must contain a list")

    remote_by_path: dict[str, Any] = {}
    for remote in inventory.files:
        if remote.path in remote_by_path:
            raise ValueError(f"inventory contains duplicate repository path: {remote.path}")
        remote_by_path[remote.path] = remote
    inventory_scenes = set(inventory.scene_hashes)
    selected_images: set[str] = set()
    seen_scenes: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("scene_name"), str):
            raise ValueError("protocol frame row has no scene_name")
        scene = row["scene_name"]
        if scene not in inventory_scenes:
            raise ValueError(f"protocol frame row has unknown scene: {scene}")
        if scene in seen_scenes:
            raise ValueError(f"duplicate protocol row for scene: {scene}")
        seen_scenes.add(scene)
        frame_paths = _load_verified_frame_paths(
            scene,
            inventory,
            data_root,
            remote_by_path,
            verified_by_repo,
            representation=representation,
        )
        scene_images: set[str] = set()
        for key in PROTOCOL_INPUT_KEYS:
            raw_values = row.get(key)
            if not isinstance(raw_values, list):
                raise ValueError(f"protocol frame row has no list for {key}: {scene}")
            for value in raw_values:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"invalid zero-based frame index in {key}: {scene}")
                if value >= len(frame_paths):
                    raise ValueError(
                        f"zero-based frame index out of range in {key}: {scene} index={value}"
                    )
                scene_images.add(f"{scene}/{frame_paths[value]}")
        # Query sampling is positional in transforms.json, just like the split inputs.
        scene_images.update(
            f"{scene}/{frame_paths[index]}"
            for index in range(0, len(frame_paths), target_every)
        )
        selected_images.update(scene_images)

    if seen_scenes != inventory_scenes:
        missing = sorted(inventory_scenes - seen_scenes)
        extra = sorted(seen_scenes - inventory_scenes)
        raise ValueError(
            f"protocol scene coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    transform_paths = {
        f"{scene}/{representation}/transforms.json" for scene in inventory.scene_hashes
    }
    required = transform_paths | selected_images
    for path in required:
        remote = remote_by_path.get(path)
        if remote is None:
            raise ValueError(f"required protocol path is absent from pinned inventory: {path}")
        if remote.kind not in {"transforms", "images_4"}:
            raise ValueError(f"required protocol path has unexpected kind: {path}")
    return required, {
        "status": "complete",
        "mapping": "transforms_frame_order",
        "protocols": list(PROTOCOL_NAMES),
        "required_files": len(required),
        "missing_files": 0,
    }


def _entry_list_sha256(entries: list[dict[str, Any]]) -> str:
    payload = "".join(
        f"{entry['repo_path']}\t{entry['bytes']}\n"
        for entry in sorted(entries, key=lambda value: value["repo_path"])
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inventory_remote_map(inventory: Any) -> dict[str, Any]:
    remote_by_path: dict[str, Any] = {}
    for remote in inventory.files:
        if remote.path in remote_by_path:
            raise ValueError(f"inventory contains duplicate repository path: {remote.path}")
        remote_by_path[remote.path] = remote
    return remote_by_path


def _verify_pair(pair: tuple[Any, Path]) -> tuple[str, dict[str, Any]]:
    remote, path = pair
    return remote.path, _verify_remote(remote, path)


def run(args: argparse.Namespace) -> int:
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be between 1 and 16")
    data_root = args.data_root.resolve()
    inventory_dir = args.inventory_dir.resolve()
    protocol_sha256 = hashlib.sha256(args.protocol_json.read_bytes()).hexdigest()
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "protocol JSON SHA256 does not match the approved pinned protocol: "
            f"{protocol_sha256}"
        )
    inventory = load_inventory_artifacts(inventory_dir)
    remote_by_path = _inventory_remote_map(inventory)
    verified_by_repo: dict[str, dict[str, Any]] = {}
    required_repo_paths, protocol_coverage = _build_required_paths(
        args.protocol_json,
        inventory,
        data_root,
        verified_by_repo,
        target_every=args.target_every,
        representation=DEFAULT_REPRESENTATION,
    )

    required_remotes = [remote_by_path[path] for path in sorted(required_repo_paths)]
    required_destinations = {
        remote.path: _destination(data_root, remote) for remote in required_remotes
    }
    expected_required_paths = {
        path.resolve() for path in required_destinations.values()
    }
    actual_paths = {
        path.resolve()
        for path in data_root.rglob("*")
        if path.is_file()
    }
    missing_paths = expected_required_paths - actual_paths
    if missing_paths:
        raise ValueError(
            "required protocol files are missing from the data root: "
            f"missing={len(missing_paths)}"
        )

    remaining_required = [
        (remote, required_destinations[remote.path])
        for remote in required_remotes
        if remote.path not in verified_by_repo
    ]
    if remaining_required:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(remaining_required))) as pool:
            for repo_path, entry in pool.map(_verify_pair, remaining_required):
                verified_by_repo[repo_path] = entry
    for remote in required_remotes:
        entry = verified_by_repo[remote.path]
        entry["relative_path"] = _relative_to_root(data_root, required_destinations[remote.path])
    entries = [verified_by_repo[remote.path] for remote in required_remotes]

    if len(entries) != EXPECTED_FILE_COUNT:
        raise ValueError(f"approved selection count changed: {len(entries)}")
    total_bytes = sum(entry["bytes"] for entry in entries)
    if total_bytes != EXPECTED_TOTAL_BYTES:
        raise ValueError(f"approved selection byte total changed: {total_bytes}")

    local_to_remote: dict[Path, Any] = {}
    for remote in inventory.files:
        if remote.scene_hash is None:
            continue
        local_path = _destination(data_root, remote).resolve()
        if local_path in local_to_remote and local_to_remote[local_path].path != remote.path:
            raise ValueError(f"inventory has duplicate local destinations: {local_path}")
        local_to_remote[local_path] = remote
    extra_paths = actual_paths - expected_required_paths
    unknown_extra = sorted(str(path) for path in extra_paths if path not in local_to_remote)
    if unknown_extra:
        raise ValueError(
            "DL3DV data root contains files outside the pinned inventory: "
            f"unexpected={len(unknown_extra)}"
        )
    extra_pairs = [
        (local_to_remote[path], path)
        for path in sorted(extra_paths, key=str)
    ]
    extra_by_repo: dict[str, dict[str, Any]] = {}
    if extra_pairs:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(extra_pairs))) as pool:
            for repo_path, entry in pool.map(_verify_pair, extra_pairs):
                entry["relative_path"] = _relative_to_root(data_root, Path(entry["relative_path"]))
                extra_by_repo[repo_path] = entry
    extra_entries = [extra_by_repo[path] for path in sorted(extra_by_repo)]
    retained_extra_bytes = sum(entry["bytes"] for entry in extra_entries)
    if len(extra_entries) != EXPECTED_RETAINED_EXTRA_FILES:
        raise ValueError(f"retained extra file count changed: {len(extra_entries)}")
    if retained_extra_bytes != EXPECTED_RETAINED_EXTRA_BYTES:
        raise ValueError(f"retained extra byte total changed: {retained_extra_bytes}")
    physical_count = len(actual_paths)
    physical_bytes = total_bytes + retained_extra_bytes
    if physical_count != EXPECTED_PHYSICAL_FILE_COUNT:
        raise ValueError(f"physical file count changed: {physical_count}")
    if physical_bytes != EXPECTED_PHYSICAL_BYTES:
        raise ValueError(f"physical byte total changed: {physical_bytes}")

    metadata_path = inventory_dir / METADATA_PATH
    if not metadata_path.is_file():
        raise ValueError(f"missing local metadata file: {metadata_path}")
    metadata_size, metadata_sha256, metadata_git_sha1 = _hash_local_file(metadata_path)
    if metadata_size != inventory.metadata_file.size:
        raise ValueError("local benchmark metadata size differs from pinned metadata")
    if inventory.metadata_file.git_blob_id is None:
        raise ValueError("pinned benchmark metadata has no Git blob identity")
    if metadata_git_sha1 != inventory.metadata_file.git_blob_id:
        raise ValueError("local benchmark metadata Git blob identity mismatch")
    manifest = json.loads((inventory_dir / "manifest.json").read_text(encoding="utf-8"))
    if metadata_sha256 != manifest["metadata_sha256"]:
        raise ValueError("local benchmark metadata differs from the pinned inventory snapshot")

    status_path = inventory_dir / "status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    if status_payload.get("status") != "complete":
        raise ValueError(f"download status is not complete: {status_payload.get('status')!r}")
    status_entries = status_payload.get("entries")
    if not isinstance(status_entries, list):
        raise ValueError("download status has no entries list")
    status_by_repo: dict[str, dict[str, Any]] = {}
    for status_entry in status_entries:
        repo_path = status_entry.get("repo_path")
        if not isinstance(repo_path, str) or repo_path in status_by_repo:
            raise ValueError("download status entries are missing or duplicate repo paths")
        status_by_repo[repo_path] = status_entry
    independent_by_repo = {entry["repo_path"]: entry for entry in entries}
    status_repo_paths = set(status_by_repo)
    if status_repo_paths != required_repo_paths:
        raise ValueError(
            "download status entries must cover every required file: "
            f"expected={len(required_repo_paths)}, actual={len(status_repo_paths)}"
        )
    for repo_path, status_entry in status_by_repo.items():
        entry = independent_by_repo[repo_path]
        if status_entry.get("status") not in {"downloaded", "reused"}:
            raise ValueError(f"download status is not successful: {repo_path}")
        if status_entry.get("sha256") != entry["local_sha256"]:
            raise ValueError(f"status SHA256 differs from independent hash: {repo_path}")
        if status_entry.get("expected_bytes", entry["bytes"]) != entry["bytes"]:
            raise ValueError(f"status size differs from independent size: {repo_path}")

    counts: dict[str, int] = {}
    for remote in required_remotes:
        counts[remote.kind] = counts.get(remote.kind, 0) + 1
    source_kinds: dict[str, int] = {}
    all_entries = entries + extra_entries
    for entry in all_entries:
        kind = str(entry["source_hash_kind"])
        source_kinds[kind] = source_kinds.get(kind, 0) + 1
    protocol_coverage["required_files"] = len(entries)
    protocol_coverage["missing_files"] = 0
    payload = {
        "schema": "mcss.dl3dv.completion_verification.v1",
        "status": "verified",
        "verified_unix": time.time(),
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": REVISION,
        "inventory_dir": str(inventory_dir),
        "protocol_json": str(args.protocol_json.resolve()),
        "protocol_sha256": protocol_sha256,
        "data_root": str(data_root),
        "representation": DEFAULT_REPRESENTATION,
        "scene_count": len(inventory.scenes),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "retained_extra_files": len(extra_entries),
        "retained_extra_bytes": retained_extra_bytes,
        "physical_count": physical_count,
        "physical_bytes": physical_bytes,
        "entry_list_sha256": _entry_list_sha256(entries),
        "entry_count_by_kind": counts,
        "protocol_coverage": protocol_coverage,
        "download_status": {
            "path": str(status_path.resolve()),
            "status": status_payload["status"],
            "entry_count": len(status_entries),
            "entry_sha256_matches": True,
            "required_entry_count": len(entries),
        },
        "source_hash_verified": {
            "all_entries": all(entry["source_hash_verified"] for entry in all_entries),
            "entry_count": len(entries),
            "extra_entry_count": len(extra_entries),
            "physical_file_count": physical_count,
            "source_hash_kind_counts": source_kinds,
            "meaning": "Each local file matched the pinned Hub LFS SHA256 or Git blob SHA1.",
        },
        "local_only": {
            "field": "entries[*].local_sha256",
            "algorithm": "sha256",
            "meaning": "Local content digest for reproducibility; not an official source checksum.",
        },
        "metadata": {
            "repo_path": METADATA_PATH,
            "bytes": metadata_size,
            "source_hash_kind": "git_blob_sha1",
            "source_hash": inventory.metadata_file.git_blob_id,
            "source_hash_verified": True,
            "local_sha256": metadata_sha256,
            "local_sha256_role": "local_only",
            "manifest_metadata_sha256": manifest["metadata_sha256"],
            "manifest_metadata_sha256_match": True,
        },
        "entries": entries,
        "extra_entries": extra_entries,
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "revision": payload["revision"],
                "file_count": payload["file_count"],
                "total_bytes": payload["total_bytes"],
                "retained_extra_files": payload["retained_extra_files"],
                "retained_extra_bytes": payload["retained_extra_bytes"],
                "physical_count": payload["physical_count"],
                "physical_bytes": payload["physical_bytes"],
                "entry_list_sha256": payload["entry_list_sha256"],
                "data_root": payload["data_root"],
                "protocol_sha256": payload["protocol_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
