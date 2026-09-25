import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mcss.v8_geometry import V8Proposal, depth_z_to_points_cam


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "run_v8_q0.py"
    spec = importlib.util.spec_from_file_location("run_v8_q0", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proposal(depth: float = 2.0) -> V8Proposal:
    intrinsics = np.asarray(
        [[4.0, 0.0, 1.0], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    depth_z = np.full((1, 2, 3), depth, dtype=np.float32)
    return V8Proposal(
        supplier="supplier",
        frame_ids=(7,),
        depth_z_m=depth_z,
        points_cam_m=depth_z_to_points_cam(depth_z[0], intrinsics)[None],
        native_confidence=np.ones_like(depth_z),
        native_risk=np.zeros_like(depth_z),
        valid_mask=np.ones_like(depth_z, dtype=np.bool_),
        intrinsics=intrinsics[None],
        c2w=np.eye(4, dtype=np.float32)[None],
    )


def test_posthoc_depth_check_converts_context_ray_distance_to_z_depth() -> None:
    runner = _load_runner()
    proposal = _proposal()
    ray_distance = np.linalg.norm(proposal.points_cam_m, axis=-1)
    result = runner._posthoc_depth_check(proposal, proposal, ray_distance)
    assert result["context_depth_semantics"] == "ray_distance_m_converted_to_opencv_z_m"
    assert result["unidepth_median_abs_error_m"] == pytest.approx(0.0, abs=1e-6)
    assert result["emvsnet_median_abs_error_m"] == pytest.approx(0.0, abs=1e-6)


def test_proposal_check_rejects_request_camera_mismatch() -> None:
    runner = _load_runner()
    proposal = _proposal()
    request = runner.GeometrySupplierRequest(
        (7,),
        runner.torch.zeros(1, 3, 2, 3),
        runner.Cameras(
            runner.torch.tensor(proposal.intrinsics),
            runner.torch.tensor(proposal.c2w),
            (2, 3),
        ),
    )
    proposal.intrinsics[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="intrinsics"):
        runner._check_proposal(proposal, request)


def test_unverified_checkpoint_training_method_is_a_hard_stop() -> None:
    runner = _load_runner()
    provenance = runner._checkpoint_method_provenance(
        {"epoch": 9, "model": {}, "best_val_mae": 3.63}, requested_method="der"
    )
    assert provenance["status"] == "FAIL"
    assert provenance["reason"] == "training_method_unverified"
    with pytest.raises(RuntimeError, match="training_method_unverified"):
        runner._require_checkpoint_method_provenance(provenance)


def test_publisher_release_provenance_requires_matching_artifact_hashes(tmp_path: Path) -> None:
    runner = _load_runner()
    checkpoint = tmp_path / "best_model_64.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    model_card = tmp_path / "README.md"
    model_card.write_text(
        "## Download Checkpoints\nbest_model_64.ckpt\n"
        "python eval.py --evidential_method der --loadckpt best_model_64.ckpt\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "provenance.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "mcss.emvsnet_checkpoint_provenance.v1",
                "checkpoint": {
                    "filename": checkpoint.name,
                    "sha256": runner._sha256_file(checkpoint),
                    "hub_model_id": "BuTTerK3ks/EMVSNet",
                    "hub_revision": "7596a73607b30771df1b6bcdb1d3e4ce2f17eef3",
                },
                "training": {
                    "evidential_method": "der",
                    "evidence_kind": "publisher_release_inference_contract",
                },
                "model_card": {
                    "sha256": runner._sha256_file(model_card),
                    "required_fragments": [
                        "best_model_64.ckpt",
                        "--evidential_method der",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    provenance = runner._checkpoint_method_provenance(
        {"epoch": 9, "model": {}},
        requested_method="der",
        checkpoint_path=checkpoint,
        provenance_path=manifest,
        model_card_path=model_card,
    )

    assert provenance["status"] == "PASS"
    assert provenance["evidence_kind"] == "publisher_release_inference_contract"
    assert provenance["checkpoint_internal_method"] is None

    checkpoint.write_bytes(b"changed")
    mismatch = runner._checkpoint_method_provenance(
        {"epoch": 9, "model": {}},
        requested_method="der",
        checkpoint_path=checkpoint,
        provenance_path=manifest,
        model_card_path=model_card,
    )
    assert mismatch["status"] == "FAIL"
    assert mismatch["reason"] == "checkpoint_sha256_mismatch"


def test_publisher_der_contract_keeps_low_alpha_optimizer_pattern_as_warning(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    checkpoint = tmp_path / "best_model_64.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    model_card = tmp_path / "README.md"
    model_card.write_text("--evidential_method der\n", encoding="utf-8")
    manifest = tmp_path / "provenance.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "mcss.emvsnet_checkpoint_provenance.v1",
                "checkpoint": {
                    "filename": checkpoint.name,
                    "sha256": runner._sha256_file(checkpoint),
                },
                "training": {
                    "evidential_method": "der",
                    "evidence_kind": "publisher_release_inference_contract",
                },
                "model_card": {
                    "sha256": runner._sha256_file(model_card),
                    "required_fragments": ["--evidential_method der"],
                },
            }
        ),
        encoding="utf-8",
    )
    active = torch.tensor([1.0, 1.0, 1e-14, 1.0]).reshape(4, 1, 1, 1, 1)
    state = {
        "model": {},
        "optimizer": {
            "param_groups": [{"params": [0, 1]}],
            "state": {
                0: {"exp_avg_sq": active.clone()},
                1: {"exp_avg_sq": active.clone()},
            },
        },
    }

    provenance = runner._checkpoint_method_provenance(
        state,
        requested_method="der",
        checkpoint_path=checkpoint,
        provenance_path=manifest,
        model_card_path=model_card,
        parameter_names=[
            "evidential.classif0.2.weight",
            "evidential.classif1.2.weight",
        ],
    )

    assert provenance["status"] == "PASS"
    assert provenance["reason"] == "inference_method_verified_by_publisher_release_contract"
    assert provenance["checkpoint_internal_method"] is None
    signature = provenance["optimizer_signature"]
    assert signature["pattern"] == "low_alpha_second_moment"
    assert signature["method_inference"] == "INDETERMINATE"
    assert signature["hard_gate_eligible"] is False
    assert "suggested_method" not in signature
    assert provenance["warnings"][0]["code"] == "optimizer_alpha_moment_imbalance"


def test_q0_report_surfaces_checkpoint_diagnostic_warnings() -> None:
    runner = _load_runner()
    record = {
        "schema_version": runner.Q0_SCHEMA_VERSION,
        "status": "GO_TO_Q1_LABEL_BLIND_AUDIT",
        "checks": {
            "checkpoint_method_provenance": {
                "status": "PASS",
                "warnings": [
                    {
                        "code": "optimizer_alpha_moment_imbalance",
                        "severity": "WARNING",
                        "message": "Adam moments cannot identify the training method.",
                    }
                ],
            }
        },
    }

    report = runner._render_report(record)

    assert "## Diagnostic Warnings" in report
    assert "optimizer_alpha_moment_imbalance" in report
    assert "Adam moments cannot identify the training method." in report


def test_repeat_stability_metrics_require_both_suppliers() -> None:
    runner = _load_runner()
    stable = _proposal(2.0)
    changed = _proposal(2.1)
    metrics = runner._repeat_stability_metrics(stable, stable, stable, changed)
    assert metrics["max_unidepth_depth_delta_m"] == pytest.approx(0.0)
    assert metrics["max_emvsnet_depth_delta_m"] == pytest.approx(0.1)
    assert metrics["passed"] is False


def test_source_drop_intervention_delta_is_computed_on_reference_overlap() -> None:
    runner = _load_runner()
    full = _proposal(2.0)
    dropped = _proposal(2.25)
    metrics = runner._source_drop_intervention_delta(full, dropped)
    assert metrics["reference_frame_id"] == 7
    assert metrics["median_depth_delta_m"] == pytest.approx(0.25)
    assert metrics["max_depth_delta_m"] == pytest.approx(0.25)


def test_determinism_configuration_is_explicit() -> None:
    runner = _load_runner()
    config = runner._configure_determinism()
    assert config["torch_deterministic_algorithms"] is True
    assert config["cudnn_deterministic"] is True
    assert config["cudnn_benchmark"] is False


def test_q0_record_separates_label_blind_proposals_from_postseal_audit(tmp_path: Path) -> None:
    runner = _load_runner()
    args = SimpleNamespace(
        window_file=tmp_path / "window.json",
        unidepth_code_root=tmp_path / "unidepth",
        emvsnet_code_root=tmp_path / "emvsnet",
        unidepth_bundle=tmp_path / "unidepth_bundle",
        emvsnet_checkpoint=tmp_path / "model.ckpt",
        emvsnet_provenance=None,
        emvsnet_model_card=None,
        device="cpu",
        evidential_method="der",
    )

    boundary = runner._initial_record(args)["research_boundary"]

    assert boundary["proposal_phase_target_access"] is False
    assert boundary["postseal_context_depth_audit"] is True
    assert "target_access" not in boundary

    record = runner._initial_record(args)
    record["status"] = "STOP_EMVSNET_Q0"
    report = runner._render_report(record)
    assert "proposal generation is label blind" in report
    assert "post-seal development context-depth audit" in report
    assert "no training, target access" not in report
