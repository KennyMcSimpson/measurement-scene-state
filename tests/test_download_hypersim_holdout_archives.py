import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from requests.structures import CaseInsensitiveDict

SPEC = importlib.util.spec_from_file_location(
    "download_hypersim_holdout_archives",
    Path(__file__).resolve().parents[1] / "scripts/download_hypersim_holdout_archives.py",
)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def _remote(expected: int = 8) -> dict[str, object]:
    return {
        "url": "https://example.test/scene.zip",
        "resolved_url": "https://cdn.example.test/scene.zip",
        "etag": '"etag-1"',
        "last_modified": "Tue, 29 Dec 2020 01:25:55 GMT",
        "expected_bytes": expected,
    }


def test_resume_identity_binds_size_and_last_modified_fallback() -> None:
    remote = _remote()
    old_source = {
        "url": remote["url"],
        "etag": None,
        "last_modified": remote["last_modified"],
        "expected_bytes": remote["expected_bytes"],
    }
    remote_without_etag = dict(remote, etag=None)

    download._validate_provenance(old_source, remote_without_etag, "manifest-hash")
    assert download._range_headers(3, 8, remote_without_etag) == {
        "Range": "bytes=3-7",
        "If-Range": remote["last_modified"],
    }

    with pytest.raises(ValueError, match="byte count"):
        download._validate_provenance(
            dict(old_source, expected_bytes=9), remote_without_etag, "manifest-hash"
        )


def test_get_response_must_match_head_identity_and_range() -> None:
    remote = _remote()
    response = SimpleNamespace(
        url=remote["resolved_url"],
        status_code=206,
        headers=CaseInsensitiveDict(
            {
                "ETag": remote["etag"],
                "Last-Modified": remote["last_modified"],
                "Content-Range": "bytes 3-7/8",
                "Content-Length": "5",
            }
        ),
    )
    download._validate_get_response(response, remote, expected=8, offset=3)

    response.headers["ETag"] = '"different"'
    with pytest.raises(ValueError, match="GET etag"):
        download._validate_get_response(response, remote, expected=8, offset=3)


def test_existing_status_hash_is_checked_and_completion_records_local_hash(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "source_manifest_sha256": "manifest-hash",
                "entries": [{"scene": "ai_001_001", "sha256": "a" * 64}],
            }
        ),
        encoding="utf-8",
    )
    hashes = download._load_status_hashes(status_path, "manifest-hash")
    assert hashes == {"ai_001_001": "a" * 64}
    with pytest.raises(ValueError, match="different source manifest"):
        download._load_status_hashes(status_path, "different-manifest")
    download._validate_local_hash({}, "a" * 64, hashes["ai_001_001"])
    with pytest.raises(ValueError, match="existing status"):
        download._validate_local_hash({}, "b" * 64, hashes["ai_001_001"])

    verification = {
        "sha256": "a" * 64,
        "zip_members": 1,
        "verification": "local SHA256",
    }
    completed = download._completion_provenance(
        {
            "url": _remote()["url"],
            "etag": _remote()["etag"],
            "last_modified": _remote()["last_modified"],
            "expected_bytes": 8,
        },
        _remote(),
        "manifest-hash",
        verification,
    )
    assert completed["local_sha256"] == "a" * 64
    assert completed["source_manifest_sha256"] == "manifest-hash"


def test_run_retries_request_failure_after_source_recorded(tmp_path: Path, monkeypatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.txt", "synthetic download fixture")
    payload = buffer.getvalue()
    rows = [
        {"scene": f"ai_001_{index:03}", "zip_bytes": len(payload), "status": "index_read_ok"}
        for index in range(1, 27)
    ]
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"verified_scenes": 26, "results": rows}), encoding="utf-8")
    monkeypatch.setattr(download, "PROJECT", tmp_path)
    monkeypatch.setattr(download, "SOURCE", source)
    monkeypatch.setattr(download, "DESTINATION", tmp_path / "data" / "archives")
    monkeypatch.setattr(download, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    calls = {}

    class Response:
        def __init__(self, url, status_code=200):
            self.url = url
            self.status_code = status_code
            self.headers = CaseInsensitiveDict({
                "Content-Length": str(len(payload)),
                "ETag": '"fixture"',
                "Last-Modified": "Tue, 29 Dec 2020 01:25:55 GMT",
            })
            if status_code == 206:
                self.headers["Content-Range"] = f"bytes 0-{len(payload)-1}/{len(payload)}"

        def raise_for_status(self):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def iter_content(self, chunk_size):
            yield payload

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def head(self, url, **kwargs):
            return Response(url)

        def get(self, url, **kwargs):
            calls[url] = calls.get(url, 0) + 1
            if url.endswith("ai_001_001.zip") and calls[url] == 1:
                raise requests.ConnectionError("synthetic failure before partial creation")
            return Response(url, 206)

    monkeypatch.setattr(download.requests, "Session", Session)
    assert download.run(workers=1) == 0
    status = json.loads((download.LOG_ROOT / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["completed"] == 26
    assert status["errors"] == []
    assert status["entries"][0]["attempts"] == 2
    assert len(list(download.DESTINATION.glob("*.zip"))) == 26
