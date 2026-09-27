import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_restore_checks_hashes_and_preserves_modified_file(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/restore_report_artifacts.py"
    spec = importlib.util.spec_from_file_location("restore_reports", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "docs/experiments/fixture/raw.json"
    target.parent.mkdir(parents=True)
    data = b'{"rows": [1, 2, 3]}\n'
    archive = target.with_suffix(".json.gz")
    archive.write_bytes(gzip.compress(data, mtime=0))
    record = {
        "path": str(target.relative_to(tmp_path)),
        "archive": str(archive.relative_to(tmp_path)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "compressed_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (tmp_path / "docs/report_artifacts.json").write_text(json.dumps({"artifacts": [record]}))
    module.restore(tmp_path, check=True)
    assert not target.exists()
    module.restore(tmp_path)
    assert target.read_bytes() == data
    target.write_text("local edit")
    with pytest.raises(FileExistsError):
        module.restore(tmp_path)
    assert target.read_text() == "local edit"
    archive.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Archive checksum"):
        module.restore(tmp_path)
