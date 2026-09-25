import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mcss.geometry_supplier import (
    CodeDependency,
    GeometrySupplierRequest,
    SupplierIdentity,
    SupplierRunConfig,
)
from mcss.mapanything_supplier import (
    MapAnythingSupplier,
    local_only_dinov2_hub,
    tracked_source_sha256,
    verify_code_dependency,
)
from mcss.types import Cameras


class _FakeBackend:
    def __init__(self) -> None:
        self.views: list[dict[str, object]] | None = None
        self.kwargs: dict[str, object] | None = None

    def preprocess(self, views: list[dict[str, object]]) -> list[dict[str, object]]:
        self.views = views
        processed = []
        for view in views:
            processed.append(
                {
                    "img": torch.zeros(1, 3, 4, 6),
                    "data_norm_type": ["dinov2"],
                    "intrinsics": torch.as_tensor(view["intrinsics"])[None],
                    "camera_poses": torch.as_tensor(view["camera_poses"])[None],
                    "is_metric_scale": torch.tensor([True]),
                }
            )
        return processed

    def infer(
        self, views: list[dict[str, object]], **kwargs: object
    ) -> list[dict[str, torch.Tensor]]:
        self.kwargs = kwargs
        predictions = []
        for index, _view in enumerate(views):
            predictions.append(
                {
                    "depth_along_ray": torch.full((1, 4, 6, 1), float(index + 1)),
                    "conf": torch.full((1, 4, 6), 3.0),
                    "mask": torch.ones(1, 4, 6, 1, dtype=torch.bool),
                    # These intentionally wrong predictions must not enter the result.
                    "intrinsics": torch.zeros(1, 3, 3),
                    "camera_poses": torch.zeros(1, 4, 4),
                    "pts3d": torch.full((1, 4, 6, 3), 999.0),
                }
            )
        return predictions


def _request() -> GeometrySupplierRequest:
    frame_ids = (9, 3)
    rgb = torch.rand(2, 3, 4, 6)
    intrinsics = torch.tensor(
        [
            [[5.0, 0.0, 2.5], [0.0, 5.0, 1.5], [0.0, 0.0, 1.0]],
            [[6.0, 0.0, 2.5], [0.0, 6.0, 1.5], [0.0, 0.0, 1.0]],
        ]
    )
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[0, 0, 3] = 9.0
    c2w[1, 0, 3] = 3.0
    return GeometrySupplierRequest(frame_ids, rgb, Cameras(intrinsics, c2w, (4, 6)))


def test_adapter_passes_only_context_rgb_known_cameras_and_metric_flag(tmp_path: Path) -> None:
    backend = _FakeBackend()
    supplier = MapAnythingSupplier(
        identity=SupplierIdentity("model", "revision", "repo", "code"),
        run_config=SupplierRunConfig(),
        code_root=tmp_path,
        backend=backend,
    )

    result = supplier.infer(_request())

    assert result.frame_ids == (3, 9)
    assert backend.views is not None
    assert [set(view) for view in backend.views] == [
        {"img", "intrinsics", "camera_poses", "is_metric_scale"},
        {"img", "intrinsics", "camera_poses", "is_metric_scale"},
    ]
    assert all(bool(view["is_metric_scale"]) for view in backend.views)
    assert np.allclose(result.intrinsics[0], _request().cameras.intrinsics[1].numpy())
    assert np.allclose(result.c2w[0], _request().cameras.c2w[1].numpy())
    assert backend.kwargs == {
        "memory_efficient_inference": True,
        "minibatch_size": 1,
        "use_amp": True,
        "amp_dtype": "bf16",
        "apply_mask": True,
        "mask_edges": True,
        "apply_confidence_mask": False,
        "use_multiview_confidence": False,
    }


def test_lazy_backend_is_loaded_only_once(tmp_path: Path) -> None:
    supplier = MapAnythingSupplier(
        identity=SupplierIdentity("model", "revision", "repo", "code"),
        run_config=SupplierRunConfig(),
        code_root=tmp_path,
        backend=None,
    )
    backend = _FakeBackend()
    calls = []

    def load_backend():
        calls.append(1)
        return backend

    supplier._load_backend = load_backend  # type: ignore[method-assign]
    supplier.infer(_request())
    supplier.infer(_request())

    assert len(calls) == 1


def _make_git_checkout(root: Path) -> str:
    root.mkdir()
    (root / "hubconf.py").write_text("VALUE = 1\n", encoding="utf-8")
    commands = (
        ["git", "init", "--quiet"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test User"],
        ["git", "add", "hubconf.py"],
        ["git", "commit", "--quiet", "-m", "fixture"],
    )
    for command in commands:
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_code_dependency_rejects_revision_hash_and_dirty_checkout(tmp_path: Path) -> None:
    root = tmp_path / "dinov2"
    revision = _make_git_checkout(root)
    source_sha256 = tracked_source_sha256(root)
    dependency = CodeDependency(
        "dinov2", "facebookresearch/dinov2", revision, source_sha256
    )

    verified = verify_code_dependency(root, dependency)
    assert verified["revision"] == revision
    assert verified["source_sha256"] == source_sha256

    with pytest.raises(RuntimeError, match="revision mismatch"):
        verify_code_dependency(
            root,
            CodeDependency("dinov2", "facebookresearch/dinov2", "0" * 40, source_sha256),
        )
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        verify_code_dependency(
            root,
            CodeDependency("dinov2", "facebookresearch/dinov2", revision, "0" * 64),
        )

    (root / "hubconf.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        verify_code_dependency(root, dependency)


def test_dinov2_hub_is_local_only_and_forbids_independent_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_load(repo_or_dir, model, *args, **kwargs):
        calls.append((repo_or_dir, model, args, kwargs))
        return "local-model"

    monkeypatch.setattr(torch.hub, "load", fake_load)

    with local_only_dinov2_hub(tmp_path):
        loaded = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitg14",
            force_reload=False,
            pretrained=False,
        )
        assert loaded == "local-model"
        with pytest.raises(RuntimeError, match="unexpected torch.hub repository"):
            torch.hub.load("another/repository", "model", pretrained=False)
        with pytest.raises(RuntimeError, match="independent DINOv2 weights"):
            torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitg14", pretrained=True
            )
        with pytest.raises(RuntimeError, match="weight download"):
            torch.hub.load_state_dict_from_url("https://example.invalid/model.pth")

    assert calls == [
        (
            str(tmp_path.resolve()),
            "dinov2_vitg14",
            (),
            {"pretrained": False, "source": "local"},
        )
    ]
