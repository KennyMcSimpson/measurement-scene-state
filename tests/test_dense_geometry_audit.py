import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mcss.dense_geometry_audit import (
    DenseAuditDecisionThresholds,
    DenseGeometryAuditConfig,
    _evaluate_camera_shuffled_control,
    _spearman,
    decide_dense_audit,
    load_dense_audit_config,
    materialize_and_evaluate_window,
    render_dense_report,
    run_dense_audit,
    validate_development_train_root,
    write_dense_audit_finalization,
)
from mcss.dense_geometry_certificates import DenseCertificateConfig
from mcss.geometry_supplier import (
    CodeDependency,
    GeometrySupplierRequest,
    GeometrySupplierResult,
    SupplierIdentity,
    SupplierRunConfig,
)
from mcss.types import Cameras


def _passing_metrics() -> dict[str, float | int]:
    return {
        "window_count": 32,
        "certified_coverage_macro": 0.20,
        "nonzero_window_fraction": 0.90,
        "geometry_error_median_m": 0.10,
        "geometry_error_p90_m": 0.30,
        "matched_confidence_median_error_m": 0.20,
        "risk_error_spearman": 0.20,
        "permutation_max_depth_delta_m": 0.0,
        "shuffled_certified_coverage_macro": 0.05,
        "shuffled_geometry_error_median_m": 1.0,
    }


def test_dense_decision_requires_every_preregistered_gate() -> None:
    thresholds = DenseAuditDecisionThresholds()
    passing = _passing_metrics()
    assert decide_dense_audit(passing, thresholds)["status"] == "GO_TO_STATE_INTEGRATION"

    failures = {
        "window_count": 31,
        "certified_coverage_macro": 0.10,
        "nonzero_window_fraction": 0.50,
        "geometry_error_median_m": 0.30,
        "geometry_error_p90_m": 0.60,
        "matched_confidence_median_error_m": 0.05,
        "risk_error_spearman": -0.1,
        "permutation_max_depth_delta_m": 0.01,
    }
    for key, value in failures.items():
        metrics = dict(passing)
        metrics[key] = value
        expected = "INCONCLUSIVE" if key == "window_count" else "STOP_MAPANYTHING_SUPPLIER"
        assert decide_dense_audit(metrics, thresholds)["status"] == expected

    no_shuffle_degradation = dict(passing)
    no_shuffle_degradation["shuffled_certified_coverage_macro"] = 0.19
    no_shuffle_degradation["shuffled_geometry_error_median_m"] = 0.12
    assert (
        decide_dense_audit(no_shuffle_degradation, thresholds)["status"]
        == "STOP_MAPANYTHING_SUPPLIER"
    )


def test_dense_decision_is_inconclusive_when_endpoint_is_missing() -> None:
    metrics = _passing_metrics()
    del metrics["risk_error_spearman"]
    decision = decide_dense_audit(metrics, DenseAuditDecisionThresholds())
    assert decision["status"] == "INCONCLUSIVE"
    assert decision["missing"] == ["risk_error_spearman"]


def test_zero_coverage_shuffle_passes_control_without_defined_error() -> None:
    metrics = _passing_metrics()
    metrics["shuffled_certified_coverage_macro"] = 0.0
    metrics["shuffled_geometry_error_median_m"] = None

    decision = decide_dense_audit(metrics, DenseAuditDecisionThresholds())

    assert decision["status"] == "GO_TO_STATE_INTEGRATION"
    assert decision["gates"]["shuffled_control_degrades"] is True


def test_spearman_uses_average_ranks_for_ties() -> None:
    assert _spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert _spearman([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.8660254)


def test_audit_root_rejects_diagnostic_and_final_partitions(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    assert validate_development_train_root(train) == train.resolve()

    for name in ("diagnostic_test", "final_holdout", "val"):
        root = tmp_path / name
        root.mkdir()
        with pytest.raises(ValueError, match="development train"):
            validate_development_train_root(root)


def test_window_pipeline_opens_depth_only_after_both_caches_are_sealed(tmp_path: Path) -> None:
    intrinsics = torch.tensor(
        [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]] * 2
    )
    cameras = Cameras(intrinsics, torch.eye(4).repeat(2, 1, 1), (5, 5))
    request = GeometrySupplierRequest((0, 1), torch.ones(2, 3, 5, 5), cameras)
    result = GeometrySupplierResult(
        frame_ids=(0, 1),
        depth_along_ray=np.ones((2, 5, 5), dtype=np.float32) * 2.0,
        confidence=np.ones((2, 5, 5), dtype=np.float32),
        valid_mask=np.ones((2, 5, 5), dtype=np.bool_),
        intrinsics=intrinsics.numpy(),
        c2w=cameras.c2w.numpy(),
    )

    class Supplier:
        def infer(self, supplied_request):
            assert not hasattr(supplied_request, "depth")
            return result

    events = []

    def load_depth():
        assert (tmp_path / "window" / "supplier" / "metadata.json").is_file()
        assert (tmp_path / "window" / "certificates" / "metadata.json").is_file()
        events.append("depth")
        return np.ones((2, 5, 5), dtype=np.float32) * 2.0, [tmp_path / "depth.npy"]

    evaluation = materialize_and_evaluate_window(
        tmp_path / "window",
        request=request,
        supplier=Supplier(),
        identity=SupplierIdentity("model", "revision", "repo", "code"),
        run_config=SupplierRunConfig(),
        certificate_config=DenseCertificateConfig(grid_stride=2),
        depth_loader=load_depth,
    )

    assert events == ["depth"]
    assert evaluation["certificate_before_depth"] is True
    assert evaluation["geometry_error_median_m"] == pytest.approx(0.0)
    assert evaluation["accessed_depth_paths"] == [str(tmp_path / "depth.npy")]


def test_window_pipeline_reuses_hash_validated_sealed_caches_after_interruption(
    tmp_path: Path,
) -> None:
    intrinsics = torch.tensor(
        [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]] * 2
    )
    cameras = Cameras(intrinsics, torch.eye(4).repeat(2, 1, 1), (5, 5))
    request = GeometrySupplierRequest((0, 1), torch.ones(2, 3, 5, 5), cameras)
    calls = []

    class Supplier:
        def infer(self, supplied_request):
            calls.append(1)
            return GeometrySupplierResult(
                supplied_request.frame_ids,
                np.ones((2, 5, 5), dtype=np.float32) * 2.0,
                np.ones((2, 5, 5), dtype=np.float32),
                np.ones((2, 5, 5), dtype=np.bool_),
                supplied_request.cameras.intrinsics.numpy(),
                supplied_request.cameras.c2w.numpy(),
            )

    kwargs = {
        "request": request,
        "supplier": Supplier(),
        "identity": SupplierIdentity("model", "revision", "repo", "code"),
        "run_config": SupplierRunConfig(),
        "certificate_config": DenseCertificateConfig(grid_stride=2),
        "depth_loader": lambda: (
            np.ones((2, 5, 5), dtype=np.float32) * 2.0,
            [tmp_path / "depth.npy"],
        ),
    }

    materialize_and_evaluate_window(tmp_path / "window", **kwargs)
    materialize_and_evaluate_window(tmp_path / "window", **kwargs)

    assert len(calls) == 1


def test_config_parser_rejects_unknown_keys_and_loads_frozen_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        """
seed: 17
dataset_root: data/train
frozen_windows: outputs/old/frozen_windows.json
output_dir: outputs/new
context_views: 4
permutation_windows: 2
supplier:
  model_id: facebook/map-anything
  model_revision: model-revision
  code_repository: facebookresearch/map-anything
  code_revision: code-revision
  dependencies:
    - name: dinov2
      repository: facebookresearch/dinov2
      revision: dinov2-revision
      source_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  code_root: third_party/map-anything
  dinov2_code_root: third_party/dinov2
  device: cuda
  model_cache_dir: outputs/model_cache/huggingface
  memory_efficient_inference: true
  minibatch_size: 1
  use_amp: true
  amp_dtype: bf16
  apply_mask: true
  mask_edges: true
  apply_confidence_mask: false
  use_multiview_confidence: false
certificate:
  grid_stride: 4
  min_depth_m: 0.1
  max_depth_m: 20.0
  depth_absolute_tolerance_m: 0.1
  depth_relative_tolerance: 0.05
  roundtrip_threshold_px: 2.0
  surface_half_width_m: 0.05
  free_space_margin_m: 0.05
  minimum_supporting_views: 1
  maximum_conflicting_views: 0
decision:
  minimum_windows: 32
  minimum_certified_coverage: 0.15
  minimum_nonzero_window_fraction: 0.8
  maximum_median_error_m: 0.25
  maximum_p90_error_m: 0.5
  maximum_permutation_delta_m: 0.0001
  minimum_shuffled_error_ratio: 1.5
  minimum_shuffled_error_margin_m: 0.05
  maximum_shuffled_coverage_ratio: 0.5
  minimum_risk_error_spearman: 0.0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_dense_audit_config(config_path)
    assert isinstance(config, DenseGeometryAuditConfig)
    assert config.supplier_identity.model_revision == "model-revision"
    assert config.supplier_identity.dependencies == (
        CodeDependency(
            "dinov2", "facebookresearch/dinov2", "dinov2-revision", "a" * 64
        ),
    )
    assert config.dinov2_code_root == "third_party/dinov2"
    assert config.run_config.minibatch_size == 1
    assert config.certificate.grid_stride == 4

    config_path.write_text(config_path.read_text(encoding="utf-8") + "unknown: true\n")
    with pytest.raises(ValueError, match="unknown"):
        load_dense_audit_config(config_path)


def test_fake_end_to_end_runner_reuses_exact_windows_and_writes_terminal_artifacts(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data" / "train"
    scene_root = dataset_root / "scene_a"
    (scene_root / "rgb").mkdir(parents=True)
    (scene_root / "depth").mkdir()
    frames = []
    intrinsics = [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
    for frame_id in range(2):
        Image.fromarray(np.full((5, 5, 3), 64 + frame_id, dtype=np.uint8)).save(
            scene_root / "rgb" / f"{frame_id:06d}.png"
        )
        np.save(scene_root / "depth" / f"{frame_id:06d}.npy", np.ones((5, 5)) * 2.0)
        frames.append(
            {
                "frame_id": frame_id,
                "rgb": f"rgb/{frame_id:06d}.png",
                "depth": f"depth/{frame_id:06d}.npy",
                "intrinsics": intrinsics,
                "c2w": np.eye(4).tolist(),
            }
        )
    manifest = {
        "schema_version": 1,
        "scene_id": "scene_a",
        "coordinate_convention": "opencv_c2w",
        "image_size": [5, 5],
        "bounds": [[-1, -1, 0.1], [1, 1, 4]],
        "frames": frames,
    }
    manifest_path = scene_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    windows_path = tmp_path / "frozen_windows.json"
    windows_path.write_text(
        json.dumps(
            {
                "partition": "train",
                "seed": 17,
                "selection_rule": "test fixture",
                "windows": [
                    {
                        "scene_id": "scene_a",
                        "start": 0,
                        "frame_ids": [0, 1],
                        "manifest": str(manifest_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    config = DenseGeometryAuditConfig(
        seed=17,
        dataset_root=str(dataset_root),
        frozen_windows=str(windows_path),
        output_dir=str(output_dir),
        context_views=2,
        permutation_windows=1,
        supplier_identity=SupplierIdentity("model", "revision", "repo", "code"),
        run_config=SupplierRunConfig(),
        code_root=str(tmp_path),
        dinov2_code_root=str(tmp_path),
        device="cpu",
        model_cache_dir=str(tmp_path / "cache"),
        certificate=DenseCertificateConfig(grid_stride=2),
        decision=DenseAuditDecisionThresholds(minimum_windows=1),
    )

    inference_calls = []

    class Supplier:
        def infer(self, request):
            inference_calls.append(request.frame_ids)
            view_count = len(request.frame_ids)
            height, width = request.cameras.image_size
            return GeometrySupplierResult(
                request.frame_ids,
                np.ones((view_count, height, width), dtype=np.float32) * 2.0,
                np.ones((view_count, height, width), dtype=np.float32),
                np.ones((view_count, height, width), dtype=np.bool_),
                request.cameras.intrinsics.numpy(),
                request.cameras.c2w.numpy(),
            )

    result = run_dense_audit(config, supplier=Supplier())

    assert result["metrics"]["window_count"] == 1
    assert len(inference_calls) == 2
    assert result["certificate_before_depth"] is True
    assert result["final_or_diagnostic_access"] is False
    assert (output_dir / "audit_result.json").is_file()
    assert (output_dir / "window_metrics.jsonl").is_file()
    assert (output_dir / "access_log.json").is_file()
    assert (output_dir / "REPORT.md").is_file()
    assert (output_dir / "artifact_hashes.json").is_file()


def test_complete_report_exposes_runtime_model_data_and_label_provenance() -> None:
    result = {
        "status": "STOP_MAPANYTHING_SUPPLIER",
        "decision": {
            "reason": "one or more gates failed",
            "gates": {"certified_coverage": True, "median_geometry_error": False},
        },
        "metrics": {
            "window_count": 32,
            "certified_coverage_macro": 0.3,
            "nonzero_window_fraction": 1.0,
            "geometry_error_median_m": 0.6,
            "geometry_error_p90_m": 3.5,
            "matched_confidence_median_error_m": 0.45,
            "risk_error_spearman": -0.01,
            "shuffled_certified_coverage_macro": 0.07,
            "shuffled_geometry_error_median_m": 1.6,
            "permutation_max_depth_delta_m": 0.0,
        },
        "certificate_before_depth": True,
        "final_or_diagnostic_access": False,
        "supplier_identity": SupplierIdentity(
            "facebook/map-anything",
            "model-revision",
            "facebookresearch/map-anything",
            "code-revision",
            (
                CodeDependency(
                    "dinov2", "facebookresearch/dinov2", "dinov2-revision", "a" * 64
                ),
            ),
        ).to_dict(),
    }
    provenance = {
        "audit_result_sha256": "b" * 64,
        "model_files": [
            {
                "name": "model.safetensors",
                "bytes": 123,
                "sha256": "c" * 64,
            }
        ],
        "code_snapshots": [
            {
                "name": "mapanything",
                "revision": "code-revision",
                "source_sha256": "d" * 64,
            }
        ],
        "runtime": {
            "python": "3.11.14",
            "torch": "2.11.0+cu128",
            "cuda_runtime": "12.8",
            "gpu": "NVIDIA GeForce RTX 5070",
            "gpu_total_bytes": 12820480000,
            "cuda_peak_memory_bytes": 5685421568,
        },
        "disk": {
            "audit_output_bytes": 1000,
            "model_cache_bytes": 2000,
            "d_free_bytes": 3000,
        },
        "training_data_disclosure": {
            "datasets": ["Aria Synthetic Environments", "BlendedMVS"],
            "hypersim_named_in_public_config": False,
            "limitation": "No direct or indirect overlap claim is established.",
        },
        "label_access": {
            "count": 2,
            "all_after_certificate_seal": True,
            "final_or_diagnostic_access": False,
            "paths": [
                r"D:\data\train\scene\depth\000001.npy",
                r"D:\data\train\scene\depth\000002.npy",
            ],
        },
    }

    report = render_dense_report(result, provenance)

    assert "model.safetensors" in report
    assert "NVIDIA GeForce RTX 5070" in report
    assert "Aria Synthetic Environments" in report
    assert "No direct or indirect overlap claim is established." in report
    assert r"D:\data\train\scene\depth\000001.npy" in report
    assert "Formal Research Institute: not invoked" in report
    assert "Evaluation Council: not convened" in report


def test_finalization_preserves_scientific_result_and_refreshes_all_hashes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    result_path = output / "audit_result.json"
    result = {
        "status": "STOP_MAPANYTHING_SUPPLIER",
        "decision": {"reason": "failed", "gates": {}},
        "metrics": {"window_count": 32},
        "certificate_before_depth": True,
        "final_or_diagnostic_access": False,
        "supplier_identity": SupplierIdentity("model", "revision", "repo", "code").to_dict(),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result_before = result_path.read_bytes()
    provenance = {
        "audit_result_sha256": __import__("hashlib").sha256(result_before).hexdigest(),
        "model_files": [],
        "code_snapshots": [],
        "runtime": {},
        "disk": {},
        "training_data_disclosure": {
            "datasets": [],
            "hypersim_named_in_public_config": False,
            "limitation": "fixture",
        },
        "label_access": {
            "count": 0,
            "all_after_certificate_seal": True,
            "final_or_diagnostic_access": False,
            "paths": [],
        },
    }

    write_dense_audit_finalization(output, provenance)

    assert result_path.read_bytes() == result_before
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))["sha256"]
    assert "provenance.json" in hashes
    assert "REPORT.md" in hashes
    for relative, expected in hashes.items():
        actual = __import__("hashlib").sha256((output / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_camera_shuffle_control_is_scored_against_real_depth_not_a_fixed_penalty() -> None:
    intrinsics = np.repeat(
        np.asarray([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]], dtype=np.float32),
        2,
        axis=0,
    )
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    supplier = GeometrySupplierResult(
        (0, 1),
        np.ones((2, 5, 5), dtype=np.float32) * 2.0,
        np.ones((2, 5, 5), dtype=np.float32),
        np.ones((2, 5, 5), dtype=np.bool_),
        intrinsics,
        c2w,
    )
    request = GeometrySupplierRequest(
        (0, 1),
        torch.ones(2, 3, 5, 5),
        Cameras(torch.from_numpy(intrinsics), torch.from_numpy(c2w), (5, 5)),
    )
    depths = np.ones((2, 5, 5), dtype=np.float32) * 2.0

    control = _evaluate_camera_shuffled_control(
        supplier, request, depths, DenseCertificateConfig(grid_stride=2)
    )

    assert control["geometry_error_median_m"] == pytest.approx(0.0)
    assert control["certified_coverage"] == pytest.approx(1.0)
