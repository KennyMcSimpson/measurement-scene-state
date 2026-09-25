"""Dynamic contract tests for the sealed RGB-only evaluator."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from mcss.dynamic.types import SealedScene, hash_scene_state
from mcss.evaluation.sealed_rgb import (
    RGBMetric,
    RGBQuery,
    evaluate_sealed_rgb,
    official_ssim_metric,
    psnr_full_image,
)
from mcss.types import Cameras, SceneState


def test_invalid_seal_rejects_queries_before_supplier_access() -> None:
    sealed = _sealed_scene()
    sealed = replace(sealed, state_hash="poisoned")
    calls: list[str] = []
    query = _query(
        sealed,
        frame_id=3,
        camera_supplier=lambda: calls.append("camera") or _camera(),
        rgb_supplier=lambda: calls.append("rgb") or torch.zeros(3, 8, 8),
    )

    with pytest.raises(ValueError, match="state_hash"):
        evaluate_sealed_rgb(sealed, [query], _rgb_renderer)

    assert calls == []


def test_query_preprocessor_runs_after_seal_validation_and_before_suppliers() -> None:
    sealed = _sealed_scene()
    events: list[str] = []

    def preprocess(query: RGBQuery) -> RGBQuery:
        events.append("preprocess")
        return query

    query = _query(
        sealed,
        frame_id=3,
        camera_supplier=lambda: events.append("camera") or _camera(),
        rgb_supplier=lambda: events.append("rgb") or torch.zeros(3, 8, 8),
    )
    evaluate_sealed_rgb(
        sealed,
        [query],
        _rgb_renderer,
        query_preprocessor=preprocess,
    )

    assert events == ["preprocess", "camera", "rgb"]

    poisoned = replace(sealed, state_hash="poisoned")
    events.clear()
    with pytest.raises(ValueError, match="state_hash"):
        evaluate_sealed_rgb(
            poisoned,
            [query],
            _rgb_renderer,
            query_preprocessor=lambda item: events.append("preprocess") or item,
        )
    assert events == []


@pytest.mark.parametrize("kind", ["scene", "episode", "split", "vault"])
def test_identity_mismatch_rejects_before_supplier_access(kind: str) -> None:
    sealed = _sealed_scene()
    calls: list[str] = []
    values = {
        "episode": sealed.episode_id,
        "scene": sealed.scene_id,
        "split": sealed.split_id,
        "vault": sealed.query_vault_id,
    }
    values[kind] = f"other-{kind}"
    query = RGBQuery(
        3,
        values["episode"],
        values["scene"],
        values["split"],
        values["vault"],
        lambda: calls.append("camera") or _camera(),
        lambda: calls.append("rgb") or torch.zeros(3, 8, 8),
        (8, 8),
    )

    with pytest.raises(ValueError, match="does not match sealed scene"):
        evaluate_sealed_rgb(sealed, [query], _rgb_renderer)

    assert calls == []


def test_query_overlap_rejects_before_supplier_access() -> None:
    sealed = _sealed_scene()
    calls: list[str] = []
    query = _query(
        sealed,
        frame_id=1,
        camera_supplier=lambda: calls.append("camera") or _camera(),
        rgb_supplier=lambda: calls.append("rgb") or torch.zeros(3, 8, 8),
    )

    with pytest.raises(ValueError, match="disjoint"):
        evaluate_sealed_rgb(sealed, [query], _rgb_renderer)

    assert calls == []


def test_target_supplier_mutation_is_rejected_and_original_state_restored() -> None:
    sealed = _sealed_scene()
    before = hash_scene_state(sealed.scene_state)

    def mutate_target() -> torch.Tensor:
        sealed.scene_state.color.add_(1.0)
        return torch.zeros(3, 8, 8)

    query = _query(sealed, frame_id=3, rgb_supplier=mutate_target)
    with pytest.raises(RuntimeError, match="sealed scene_state"):
        evaluate_sealed_rgb(sealed, [query], _rgb_renderer)

    assert hash_scene_state(sealed.scene_state) == before


def test_anchor_is_applied_once_and_renderer_receives_rgb_only() -> None:
    sealed = _sealed_scene()
    sealed.anchor_c2w[0, 3] = 1.0
    captured: dict[str, object] = {}
    world_camera = _camera()
    world_camera.c2w[0, 3] = 3.0

    def render(state: SceneState, cameras: Cameras, measurements: set[str]):
        captured["c2w"] = cameras.c2w.detach().clone()
        captured["measurements"] = measurements
        return {"rgb": torch.zeros(1, 1, 3, 8, 8)}

    query = _query(
        sealed,
        frame_id=3,
        camera_supplier=lambda: world_camera,
        rgb_supplier=lambda: torch.zeros(3, 8, 8),
    )
    result = evaluate_sealed_rgb(sealed, [query], render)

    assert captured["measurements"] == {"rgb"}
    assert captured["c2w"][0, 0, 0, 3].item() == pytest.approx(2.0)
    assert "depth" not in result["per_image"][0]


def test_psnr_uses_exact_full_unmasked_rgb_mse() -> None:
    prediction = torch.zeros(3, 8, 8)
    target = torch.full((3, 8, 8), 0.5)

    assert psnr_full_image(prediction, target) == pytest.approx(6.0205999133)
    assert math.isinf(psnr_full_image(target, target))
    assert psnr_full_image(torch.full_like(target, 2.0), target) == pytest.approx(
        6.0205999133
    )


def test_ssim_matches_official_skimage_call() -> None:
    skimage = pytest.importorskip("skimage.metrics")
    torch.manual_seed(4)
    prediction = torch.rand(3, 8, 8)
    target = torch.rand(3, 8, 8)
    expected = skimage.structural_similarity(
        prediction.numpy(), target.numpy(), channel_axis=0, data_range=1.0
    )

    metric = official_ssim_metric()
    assert metric.evaluate(prediction, target) == pytest.approx(expected)
    assert metric.full_resolution
    assert "channel_axis=0" in metric.preprocessing


def test_custom_metric_receives_one_full_resolution_image_per_query() -> None:
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def metric(prediction: torch.Tensor, target: torch.Tensor) -> float:
        seen.append((tuple(prediction.shape), tuple(target.shape)))
        return float((prediction - target).abs().mean())

    sealed = _sealed_scene()
    queries = [_query(sealed, frame_id=3), _query(sealed, frame_id=4)]
    result = evaluate_sealed_rgb(
        sealed,
        queries,
        _rgb_renderer,
        metrics={"mae": RGBMetric("mae", metric)},
    )

    assert seen == [((3, 8, 8), (3, 8, 8)), ((3, 8, 8), (3, 8, 8))]
    assert len(result["per_image"]) == 2
    assert "averages" not in result


def _sealed_scene() -> SealedScene:
    state = SceneState(
        density_logits=torch.ones(1, 1, 2, 2, 2),
        color=torch.full((1, 3, 2, 2, 2), 0.5),
        log_variance=torch.zeros(1, 1, 2, 2, 2),
        bounds=torch.tensor([[[-1.0, -1.0, 0.1], [1.0, 1.0, 1.0]]]),
    )
    return SealedScene(
        episode_id="episode-a",
        scene_id="scene-a",
        split_id="dev",
        query_vault_id="vault-a",
        scene_state=state,
        state_hash=hash_scene_state(state),
        observed_ids=(0, 1, 2),
        checkpoint_hash="checkpoint-a",
        config_hash="config-a",
        fast_state_hash="fast-a",
        anchor_c2w=torch.eye(4),
    )


def _query(
    sealed: SealedScene,
    *,
    frame_id: int,
    camera_supplier=lambda: _camera(),
    rgb_supplier=lambda: torch.full((3, 8, 8), 0.5),
) -> RGBQuery:
    return RGBQuery.for_sealed(
        sealed,
        frame_id,
        camera_supplier,
        rgb_supplier,
        image_size=(8, 8),
    )


def _camera() -> Cameras:
    return Cameras(
        torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
        torch.eye(4),
        (8, 8),
    )


def _rgb_renderer(
    state: SceneState, cameras: Cameras, measurements: set[str] | None = None
) -> dict[str, torch.Tensor]:
    assert measurements in (None, {"rgb"})
    return {"rgb": torch.zeros(1, 1, 3, 8, 8)}
