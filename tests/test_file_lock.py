"""Cross-process exclusion and release for portable maintenance-script locks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mcss.file_lock import acquire_file_lock, release_file_lock

PROBE = """
import sys
from mcss.file_lock import acquire_file_lock, release_file_lock
with open(sys.argv[1], 'r+b') as handle:
    try:
        acquire_file_lock(handle)
    except OSError:
        sys.exit(23)
    release_file_lock(handle)
"""


def _probe(path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-c", PROBE, str(path)], env=env, timeout=10, check=False
    ).returncode


@pytest.mark.parametrize("explicit_release", [True, False])
def test_file_lock_blocks_other_process_and_releases(tmp_path, explicit_release):
    path = tmp_path / "process.lock"
    path.write_bytes(b"0")
    with path.open("r+b") as handle:
        acquire_file_lock(handle)
        assert _probe(path) == 23
        if explicit_release:
            release_file_lock(handle)
            assert _probe(path) == 0
    assert _probe(path) == 0
    assert path.read_bytes() == b"0"
