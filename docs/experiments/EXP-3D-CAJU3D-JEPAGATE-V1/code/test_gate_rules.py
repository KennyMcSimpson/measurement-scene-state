"""Decision rules and root integrity of EXP-3D-CAJU3D-JEPAGATE-V1 (CPU, repository root, PYTHONPATH=src)."""

import hashlib
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import gate_score as GS  # noqa: E402

ROOT = CODE.parent
ENGINE = ROOT.parent / "EXP-3D-CAJU3D-ENGINE-V1"


def test_go_needs_jepa_below_core_b_and_depth():
    assert GS.decide({"GATE_JEPA": 0.098, "CORE_B": 0.099, "GATE_DEPTH": 0.0985})
    assert not GS.decide({"GATE_JEPA": 0.098, "CORE_B": 0.099, "GATE_DEPTH": 0.0975})
    assert not GS.decide({"GATE_JEPA": 0.0995, "CORE_B": 0.099, "GATE_DEPTH": 0.100})


def test_branches():
    assert GS.branch({"status": "SUPPORTED"}, {"mean": 0.001}) == "JEPA_ADDED_VALUE_SUPPORTED"
    assert GS.branch({"status": "SUPPORTED"}, {"mean": -0.001}) == "JEPA_ADDED_VALUE_PARTIAL"
    assert GS.branch({"status": "PARTIAL"}, {"mean": 0.001}) == "JEPA_ADDED_VALUE_PARTIAL"
    assert GS.branch({"status": "HARMFUL"}, {"mean": -0.001}) == "JEPA_ADDED_VALUE_HARMFUL"
    assert GS.branch({"status": "NOT_ESTABLISHED"}, {"mean": 0.1}) == "JEPA_ADDED_VALUE_NOT_ESTABLISHED"


def test_root_files_are_byte_copies_of_engine_v1():
    if not (ROOT / "frame_role_lock.json").is_file():
        pytest.skip("the root is not built yet")
    for name in ("frame_role_lock.json", "reused_seals.json"):
        assert hashlib.sha256((ROOT / name).read_bytes()).digest() == hashlib.sha256((ENGINE / name).read_bytes()).digest()
