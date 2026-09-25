"""CPU-only contract checks for the synthetic DL3DV resource preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from mcss.geometry import make_intrinsics

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_dl3dv_resources.py"
_SPEC = importlib.util.spec_from_file_location("check_dl3dv_resources", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ALLOWED_INPUT_COUNTS = _MODULE.ALLOWED_INPUT_COUNTS
IMAGE_SIZE = _MODULE.IMAGE_SIZE
_new_output_dir = _MODULE._new_output_dir
_parser = _MODULE._parser
run_preflight = _MODULE.run_preflight
source_manifest = _MODULE.source_manifest
synthetic_camera = _MODULE.synthetic_camera
synthetic_rgb = _MODULE.synthetic_rgb


def test_parser_defaults_to_indexed_cuda_and_pins_input_counts() -> None:
    args = _parser().parse_args(
        ["--checkpoint", "checkpoint.pt", "--output", "out", "--input-count", "16"]
    )

    assert args.device == "cuda:0"
    assert args.input_count == 16
    assert tuple(_parser()._actions[-1].choices) == ALLOWED_INPUT_COUNTS


def test_synthetic_camera_is_deterministic_and_rigid_at_full_resolution() -> None:
    device = torch.device("cpu")
    intrinsics = make_intrinsics(IMAGE_SIZE, 60.0, device=device)

    first = synthetic_camera(7, intrinsics, device)
    second = synthetic_camera(7, intrinsics, device)
    rotation = first.c2w[:3, :3]

    assert first.image_size == IMAGE_SIZE
    assert torch.equal(first.c2w, second.c2w)
    assert torch.equal(first.intrinsics, second.intrinsics)
    assert torch.allclose(rotation.transpose(0, 1) @ rotation, torch.eye(3))
    assert abs(torch.linalg.det(rotation).item()) == pytest.approx(1.0)
    assert torch.equal(first.c2w[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_synthetic_rgb_is_deterministic_and_bounded() -> None:
    x = torch.linspace(0.0, 1.0, 9).view(1, -1).expand(7, -1)
    y = torch.linspace(0.0, 1.0, 7).view(-1, 1).expand(-1, 9)

    first = synthetic_rgb(3, x, y)
    second = synthetic_rgb(3, x, y)

    assert first.shape == (3, 7, 9)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0


def test_source_manifest_hashes_cli_and_runtime_modules() -> None:
    manifest = source_manifest()

    assert manifest["files"]
    assert any(path.endswith("check_dl3dv_resources.py") for path in manifest["files"])
    assert all(len(digest) == 64 for digest in manifest["files"].values())
    assert len(manifest["sha256"]) == 64


def test_output_directory_must_be_new(tmp_path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _new_output_dir(existing)


def test_run_refuses_existing_output_before_checkpoint_access(tmp_path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_preflight(
            tmp_path / "missing-checkpoint.pt",
            existing,
            device="cpu",
            input_count=16,
        )


def test_missing_checkpoint_keeps_failed_audit_artifacts(tmp_path) -> None:
    output = tmp_path / "preflight"
    report = run_preflight(
        tmp_path / "missing-checkpoint.pt",
        output,
        device="cpu",
        input_count=16,
    )

    assert report["status"] == "failed"
    assert report["failure"]["stage"] == "load_checkpoint"
    assert report["source"]["source_manifest_stable"] is True
    assert report["target_render"]["status"] == "not_started"
    assert (output / "report.json").is_file()
    assert (output / "history.jsonl").is_file()
