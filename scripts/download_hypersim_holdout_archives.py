"""Download the approved 26 full holdout ZIPs; never extract or evaluate their contents."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from mcss.file_lock import acquire_file_lock

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs/dataset_access_check_20260918/hypersim_holdout_download_sizes.json"
DESTINATION = PROJECT / "data/hypersim_final_holdout_archives"
LOG_ROOT = PROJECT / "outputs/hypersim_holdout_full_download_20260918"
RESERVE = 10 * 1024**3
PROVENANCE_FIELDS = ("etag", "last_modified")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_archive(path: Path, expected: int) -> dict:
    if path.stat().st_size != expected:
        raise ValueError(f"Archive byte count mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        members = len(archive.infolist())
        if not members:
            raise ValueError(f"Empty ZIP: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "sha256": digest,
        "zip_members": members,
        "verification": (
            "Expected byte count, ZIP central directory, local SHA256; "
            "no official checksum or full member CRC"
        ),
    }


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {description}: expected an object: {path}")
    return value


def _remote_identity(response: Any, url: str, expected: int) -> dict[str, Any]:
    content_length = response.headers.get("Content-Length")
    try:
        actual_length = int(content_length)
    except (TypeError, ValueError) as error:
        raise ValueError("Remote archive has no valid Content-Length") from error
    if actual_length != expected:
        raise ValueError("Remote size changed since the approved inventory")
    identity = {
        "url": url,
        "resolved_url": response.url,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "expected_bytes": expected,
    }
    if identity["etag"] is None and identity["last_modified"] is None:
        raise ValueError("Remote archive has no ETag or Last-Modified validator")
    return identity


def _fetch_remote_identity(session: requests.Session, url: str, expected: int) -> dict[str, Any]:
    response = session.head(url, allow_redirects=True, timeout=(15, 30))
    try:
        response.raise_for_status()
        return _remote_identity(response, url, expected)
    finally:
        response.close()


def _validate_provenance(
    provenance: dict[str, Any],
    remote: dict[str, Any],
    source_manifest_sha256: str,
) -> None:
    if provenance.get("url") != remote["url"]:
        raise ValueError("Source URL changed; preserving local archive")
    try:
        prior_expected = int(provenance["expected_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Provenance has no valid expected byte count") from error
    if prior_expected != remote["expected_bytes"]:
        raise ValueError("Provenance byte count changed; preserving local archive")
    for field in PROVENANCE_FIELDS:
        if field not in provenance or provenance[field] != remote[field]:
            raise ValueError(f"Source {field} changed; preserving local archive")
    prior_resolved_url = provenance.get("resolved_url")
    if prior_resolved_url is not None and prior_resolved_url != remote["resolved_url"]:
        raise ValueError("Source redirect target changed; preserving local archive")
    prior_manifest_sha256 = provenance.get("source_manifest_sha256")
    if prior_manifest_sha256 is not None and prior_manifest_sha256 != source_manifest_sha256:
        raise ValueError("Source manifest changed; preserving local archive")


def _load_status_hashes(path: Path, source_manifest_sha256: str | None = None) -> dict[str, str]:
    if not path.exists():
        return {}
    status = _read_json_object(path, "existing download status")
    prior_manifest_sha256 = status.get("source_manifest_sha256")
    if (
        source_manifest_sha256 is not None
        and prior_manifest_sha256 is not None
        and prior_manifest_sha256 != source_manifest_sha256
    ):
        raise ValueError("Existing status was created from a different source manifest")
    entries = status.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid existing download status entries: {path}")
    hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("scene"), str):
            raise ValueError(f"Invalid existing download status entry: {path}")
        digest = entry.get("sha256")
        if digest is None:
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid existing local SHA256 for {entry['scene']}: {path}")
        hashes[entry["scene"]] = digest
    return hashes


def _validate_local_hash(
    provenance: dict[str, Any], local_sha256: str, status_sha256: str | None
) -> None:
    prior_sha256 = provenance.get("local_sha256")
    if prior_sha256 is None:
        prior_sha256 = provenance.get("sha256")
    if prior_sha256 is not None and prior_sha256 != local_sha256:
        raise ValueError("Local archive SHA256 differs from provenance")
    if status_sha256 is not None and status_sha256 != local_sha256:
        raise ValueError("Local archive SHA256 differs from existing status")


def _completion_provenance(
    provenance: dict[str, Any],
    remote: dict[str, Any],
    source_manifest_sha256: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    completed = dict(provenance)
    completed.update(remote)
    completed["source_manifest_sha256"] = source_manifest_sha256
    completed["local_sha256"] = verification["sha256"]
    completed["zip_members"] = verification["zip_members"]
    completed["verification"] = verification["verification"]
    return completed


def _range_headers(offset: int, expected: int, remote: dict[str, Any]) -> dict[str, str]:
    headers = {"Range": f"bytes={offset}-{expected - 1}"}
    validator = remote["etag"] or remote["last_modified"]
    if validator:
        headers["If-Range"] = validator
    return headers


def _validate_get_response(
    response: Any, remote: dict[str, Any], expected: int, offset: int
) -> None:
    if response.url != remote["resolved_url"]:
        raise ValueError("GET redirect target differs from HEAD")
    for field in PROVENANCE_FIELDS:
        if response.headers.get(field.replace("_", "-")) != remote[field]:
            raise ValueError(f"GET {field} differs from HEAD")
    if response.status_code == 206:
        content_range = response.headers.get("Content-Range")
        expected_range = f"bytes {offset}-{expected - 1}/{expected}"
        if content_range != expected_range:
            raise ValueError(f"Unexpected Content-Range: {content_range}")
        expected_body_bytes = expected - offset
    elif response.status_code == 200 and offset == 0:
        expected_body_bytes = expected
    else:
        raise ValueError("Server did not safely honor resume request")
    content_length = response.headers.get("Content-Length")
    try:
        actual_body_bytes = int(content_length)
    except (TypeError, ValueError) as error:
        raise ValueError("GET response has no valid Content-Length") from error
    if actual_body_bytes != expected_body_bytes:
        raise ValueError("GET response length differs from the approved range")


def run(workers: int) -> int:
    if workers not in (1, 2, 3, 4):
        raise ValueError("workers must be between 1 and 4")
    source_bytes = SOURCE.read_bytes()
    source_manifest_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    rows = source["results"]
    if len(rows) != 26 or source["verified_scenes"] != 26:
        raise ValueError("Expected the approved 26-scene download manifest")
    for row in rows:
        if not re.fullmatch(r"ai_\d{3}_\d{3}", row["scene"]):
            raise ValueError("Unsafe scene ID")
        if row["status"] != "index_read_ok" or int(row["zip_bytes"]) <= 0:
            raise ValueError("Unverified source archive")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DESTINATION.resolve().relative_to(PROJECT.resolve() / "data")
    lock_file = (LOG_ROOT / "download.lock").open("a+b")
    lock_file.seek(0)
    if not lock_file.read(1):
        lock_file.write(b"0")
        lock_file.flush()
    lock_file.seek(0)
    acquire_file_lock(lock_file)
    # The open handle keeps the process lock; stale files do not block resumes.
    existing_status_hashes = _load_status_hashes(
        LOG_ROOT / "status.json", source_manifest_sha256
    )
    entries = {}
    for row in rows:
        scene = row["scene"]
        final = DESTINATION / f"{scene}.zip"
        partial = DESTINATION / f"{scene}.zip.part"
        current = (
            final.stat().st_size
            if final.exists()
            else (partial.stat().st_size if partial.exists() else 0)
        )
        entries[scene] = {
            "scene": scene,
            "expected_bytes": int(row["zip_bytes"]),
            "bytes": current,
            "status": "queued",
            "attempts": 0,
        }
        if scene in existing_status_hashes:
            entries[scene]["sha256"] = existing_status_hashes[scene]
    needed = sum(max(0, e["expected_bytes"] - e["bytes"]) for e in entries.values())
    if shutil.disk_usage(DESTINATION).free < needed + RESERVE:
        raise OSError("Insufficient disk space for approved archives plus 10 GiB reserve")
    started = time.time()
    initial_bytes = sum(e["bytes"] for e in entries.values())
    mutex = threading.Lock()
    finished = threading.Event()

    def update(scene: str, **values) -> None:
        with mutex:
            entries[scene].update(values)

    def snapshot(status: str = "running") -> dict:
        with mutex:
            current = [dict(e) for e in entries.values()]
        total = sum(e["bytes"] for e in current)
        return {
            "schema": "mcss.holdout.full_download.v1",
            "status": status,
            "pid": os.getpid(),
            "started_unix": started,
            "updated_unix": time.time(),
            "destination": str(DESTINATION),
            "workers": workers,
            "source_manifest_sha256": source_manifest_sha256,
            "expected_bytes": sum(e["expected_bytes"] for e in current),
            "downloaded_bytes": total,
            "average_bytes_per_second": (total - initial_bytes) / max(1, time.time() - started),
            "completed": sum(e["status"] == "complete" for e in current),
            "total_archives": len(current),
            "entries": current,
            "split": "final_holdout",
            "extraction_performed": False,
            "scientific_evaluation_performed": False,
        }

    def monitor() -> None:
        while not finished.is_set():
            atomic_json(LOG_ROOT / "status.json", snapshot())
            finished.wait(5)

    def download(row: dict) -> None:
        scene, expected = row["scene"], int(row["zip_bytes"])
        final = DESTINATION / f"{scene}.zip"
        partial = DESTINATION / f"{scene}.zip.part"
        provenance_path = LOG_ROOT / f"{scene}.source.json"
        url = f"https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"
        for attempt in range(1, 7):
            try:
                update(scene, status="connecting", attempts=attempt)
                with requests.Session() as session:
                    remote = _fetch_remote_identity(session, url, expected)
                    if final.exists():
                        update(scene, status="verifying")
                        provenance = _read_json_object(provenance_path, "archive provenance")
                        _validate_provenance(provenance, remote, source_manifest_sha256)
                        verification = verify_archive(final, expected)
                        _validate_local_hash(
                            provenance,
                            verification["sha256"],
                            existing_status_hashes.get(scene),
                        )
                        atomic_json(
                            provenance_path,
                            _completion_provenance(
                                provenance,
                                remote,
                                source_manifest_sha256,
                                verification,
                            ),
                        )
                        update(scene, status="complete", bytes=expected, error=None, **verification)
                        return
                    if partial.exists():
                        provenance = _read_json_object(provenance_path, "archive provenance")
                        _validate_provenance(provenance, remote, source_manifest_sha256)
                    else:
                        if provenance_path.exists():
                            # A request can fail after source recording but before opening the file.
                            provenance = _read_json_object(provenance_path, "archive provenance")
                            _validate_provenance(provenance, remote, source_manifest_sha256)
                        else:
                            provenance = {
                                **remote,
                                "source_manifest_sha256": source_manifest_sha256,
                            }
                            atomic_json(provenance_path, provenance)
                    offset = partial.stat().st_size if partial.exists() else 0
                    if offset > expected:
                        raise ValueError("Partial file is larger than the expected archive")
                    if offset < expected:
                        headers = _range_headers(offset, expected, remote)
                        with session.get(
                            url, headers=headers, stream=True, timeout=(15, 60)
                        ) as body:
                            body.raise_for_status()
                            _validate_get_response(body, remote, expected, offset)
                            update(scene, status="downloading", bytes=offset)
                            with partial.open("ab" if partial.exists() else "xb") as handle:
                                for chunk in body.iter_content(chunk_size=4 * 1024**2):
                                    if not chunk:
                                        continue
                                    if offset + len(chunk) > expected:
                                        raise ValueError("Response exceeded approved byte count")
                                    if shutil.disk_usage(DESTINATION).free < RESERVE:
                                        raise OSError("Reached 10 GiB free-space reserve")
                                    handle.write(chunk)
                                    offset += len(chunk)
                                    update(scene, bytes=offset)
                    update(scene, status="verifying")
                    verification = verify_archive(partial, expected)
                    _validate_local_hash(
                        provenance,
                        verification["sha256"],
                        existing_status_hashes.get(scene),
                    )
                    partial.rename(final)  # On Windows this refuses an existing destination.
                    atomic_json(
                        provenance_path,
                        _completion_provenance(
                            provenance,
                            remote,
                            source_manifest_sha256,
                            verification,
                        ),
                    )
                    update(scene, status="complete", bytes=expected, error=None, **verification)
                    return
            except (requests.RequestException, OSError, ValueError) as error:
                update(scene, status="retrying", error=str(error))
                if attempt < 6:
                    time.sleep(min(30, 5 * attempt))
        update(scene, status="failed")

    monitoring = threading.Thread(target=monitor, daemon=True)
    monitoring.start()
    errors = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(download, row): row["scene"] for row in rows}
            for future in concurrent.futures.as_completed(future_map):
                scene = future_map[future]
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{scene}: {error}")
                    update(scene, status="failed", error=str(error))
                print(json.dumps({"scene": scene, "status": entries[scene]["status"]}), flush=True)
    finally:
        finished.set()
        monitoring.join()
    state = snapshot(
        "complete" if all(e["status"] == "complete" for e in entries.values()) else "failed"
    )
    reported_errors = list(errors)
    reported_set = set(reported_errors)
    for entry in state["entries"]:
        if entry["status"] != "failed":
            continue
        message = f"{entry['scene']}: {entry.get('error') or 'download failed'}"
        if message not in reported_set:
            reported_errors.append(message)
            reported_set.add(message)
    state["errors"] = reported_errors
    atomic_json(LOG_ROOT / "status.json", state)
    lock_file.close()
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    raise SystemExit(run(parser.parse_args().workers))
