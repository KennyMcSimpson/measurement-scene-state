"""Regression tests for the post-seal query-vault boundary."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dynamic.types import SealedScene, hash_scene_state
from mcss.evaluation import sealed_queries
from mcss.evaluation.sealed_queries import evaluate_sealed
from mcss.measurements import FixedMeasurementRenderer
from mcss.types import SceneState

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifact_dir() -> Path:
    output_root = PROJECT / "outputs"
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test_sealed_queries_", dir=output_root) as directory:
        yield Path(directory)


def test_evaluate_sealed_reports_per_query_metrics_and_preserves_hash(artifact_dir: Path) -> None:
    sealed = _sealed_scene()
    vault_path = _write_vault(artifact_dir, sealed)
    before = hash_scene_state(sealed.scene_state)

    result = evaluate_sealed(
        sealed,
        vault_path,
        FixedMeasurementRenderer(n_samples=2, ray_chunk_size=8),
    )

    assert set(result) == {"per_query", "averages"}
    assert result["per_query"][0]["frame_id"] == 5
    assert set(result["averages"]) == {
        "rgb_mse",
        "rgb_psnr",
        "depth_abs_rel",
        "opacity_mean",
        "coverage",
    }
    assert hash_scene_state(sealed.scene_state) == before


def test_mismatched_scene_is_rejected_before_query_image_io(
    artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = _sealed_scene()
    vault_path = _write_vault(artifact_dir, sealed, scene_id="other_scene")

    def unexpected_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("query RGB must not be opened before identity validation")

    monkeypatch.setattr(sealed_queries.Image, "open", unexpected_open)
    with pytest.raises(ValueError, match="scene_id does not match"):
        evaluate_sealed(sealed, vault_path, FixedMeasurementRenderer(n_samples=2))


def test_altered_sealed_state_is_rejected_before_query_image_io(
    artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = _sealed_scene()
    vault_path = _write_vault(artifact_dir, sealed)
    sealed.scene_state.color.add_(0.1)

    def unexpected_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("query RGB must not be opened after a failed seal hash")

    monkeypatch.setattr(sealed_queries.Image, "open", unexpected_open)
    with pytest.raises(ValueError, match="state_hash"):
        evaluate_sealed(sealed, vault_path, FixedMeasurementRenderer(n_samples=2))


def test_query_camera_is_expressed_relative_to_sealed_anchor(artifact_dir: Path) -> None:
    anchor = torch.eye(4)
    anchor[0, 3] = 1.0
    sealed = _sealed_scene(anchor_c2w=anchor)
    vault_path = _write_vault(artifact_dir, sealed, query_x=2.0)
    measurement = _CaptureMeasurement()

    evaluate_sealed(sealed, vault_path, measurement)

    assert measurement.c2w is not None
    assert measurement.c2w[0, 0, 0, 3].item() == pytest.approx(1.0)


def test_measurement_state_mutation_is_detected_after_query_io(artifact_dir: Path) -> None:
    sealed = _sealed_scene()
    vault_path = _write_vault(artifact_dir, sealed)

    with pytest.raises(RuntimeError, match="modified during query evaluation"):
        evaluate_sealed(sealed, vault_path, _MutatingMeasurement())


def _sealed_scene(*, anchor_c2w: torch.Tensor | None = None) -> SealedScene:
    bounds = torch.tensor([[[-1.0, -1.0, 0.1], [1.0, 1.0, 1.0]]])
    state = SceneState(
        density_logits=torch.full((1, 1, 2, 2, 2), 2.0),
        color=torch.full((1, 3, 2, 2, 2), 0.5),
        log_variance=torch.zeros((1, 1, 2, 2, 2)),
        bounds=bounds,
    )
    return SealedScene(
        episode_id="dev--scene_a--cam_00",
        scene_id="scene_a",
        split_id="dev",
        query_vault_id="qv-test",
        scene_state=state,
        state_hash=hash_scene_state(state),
        observed_ids=(0, 1, 2),
        checkpoint_hash="checkpoint",
        config_hash="config",
        fast_state_hash="fast",
        anchor_c2w=torch.eye(4) if anchor_c2w is None else anchor_c2w,
    )


def _write_vault(
    directory: Path,
    sealed: SealedScene,
    *,
    scene_id: str | None = None,
    query_x: float = 0.0,
) -> Path:
    rgb_path = directory / "query.png"
    depth_path = directory / "query.npy"
    Image.new("RGB", (2, 2), (128, 128, 128)).save(rgb_path)
    np.save(depth_path, np.ones((2, 2), dtype=np.float32))
    c2w = torch.eye(4)
    c2w[0, 3] = query_x
    vault = {
        "schema_version": "mcss.dynamic.query_vault.v1",
        "episodes": [
            {
                "episode_id": sealed.episode_id,
                "scene_id": sealed.scene_id if scene_id is None else scene_id,
                "split_id": sealed.split_id,
                "query_vault_id": sealed.query_vault_id,
                "image_size": [2, 2],
                "query": [
                    {
                        "frame_id": 5,
                        "intrinsics": [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
                        "c2w": c2w.tolist(),
                        "rgb": str(rgb_path),
                        "depth": str(depth_path),
                    }
                ],
            }
        ],
    }
    path = directory / "vault.json"
    path.write_text(json.dumps(vault), encoding="utf-8")
    return path


class _CaptureMeasurement:
    def __init__(self) -> None:
        self.c2w: torch.Tensor | None = None

    def __call__(
        self, state: SceneState, cameras, _measurements: set[str]
    ) -> dict[str, torch.Tensor]:
        self.c2w = cameras.c2w.clone()
        height, width = cameras.image_size
        shape = (1, 1, 1, height, width)
        return {
            "rgb": torch.zeros((1, 1, 3, height, width), device=state.color.device),
            "depth": torch.ones(shape, device=state.color.device),
            "visibility": torch.ones(shape, device=state.color.device),
        }


class _MutatingMeasurement(_CaptureMeasurement):
    def __call__(
        self, state: SceneState, cameras, measurements: set[str]
    ) -> dict[str, torch.Tensor]:
        state.color.add_(0.1)
        return super().__call__(state, cameras, measurements)
