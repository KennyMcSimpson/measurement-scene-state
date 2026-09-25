"""Context-only telemetry for fixed V6 depth-candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from mcss.config import load_config
from mcss.data.collate import collate_scene_examples
from mcss.engine import Trainer
from mcss.model.evidence_encoder import (
    DualEvidenceStateEncoder,
    TransportEvidenceStateEncoder,
    _surface_peakness_with_telemetry,
    _tensor_summary,
    _transport_surface_evidence,
)


@torch.no_grad()
def run_evidence_telemetry(
    config_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    dataset_index: int = 0,
) -> dict[str, object]:
    """Replay one state and persist target-label-free candidate diagnostics."""

    config_path = Path(config_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    report_path = output_dir / "evidence_telemetry.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing telemetry report: {report_path}")
    config = load_config(config_path)
    if config.model.state_architecture not in {
        "dual_evidence",
        "dual_evidence_transport",
    }:
        raise ValueError(
            "evidence telemetry requires state_architecture: dual_evidence or "
            "dual_evidence_transport"
        )
    trainer = Trainer(config)
    if not 0 <= dataset_index < len(trainer.dataset):
        raise IndexError("dataset_index is outside the configured dataset")
    trainer.load_checkpoint(checkpoint, load_optimizer=False)
    trainer.model.eval()
    batch = collate_scene_examples([trainer.dataset[dataset_index]]).to(trainer.device)
    with torch.autocast(
        device_type=trainer.device.type,
        dtype=trainer.amp_dtype,
        enabled=trainer.amp_enabled,
    ):
        output = trainer.model(
            batch.context_rgb,
            batch.context_cameras,
            batch.target_cameras,
            batch.bounds,
        )
    dual = output.state.dual_evidence
    transport = output.state.transport_evidence
    if config.model.state_architecture == "dual_evidence":
        if dual is None or not isinstance(trainer.model.state_encoder, DualEvidenceStateEncoder):
            raise RuntimeError("checkpoint did not produce a dual-evidence state")
        evidence = dual
    else:
        if transport is None or not isinstance(
            trainer.model.state_encoder, TransportEvidenceStateEncoder
        ):
            raise RuntimeError("checkpoint did not produce a transport-evidence state")
        evidence = transport

    replay_peakness, candidate_profiles = _surface_peakness_with_telemetry(
        batch.context_rgb,
        batch.context_cameras,
        batch.bounds,
        config.model.voxel_resolution,
        evidence_temperature=config.model.evidence_temperature,
        surface_peak_temperature=config.model.surface_peak_temperature,
        chunk_size=trainer.model.state_encoder.surface_peak_chunk_size,
        appearance_confidence=evidence.appearance_confidence,
    )
    state_report: dict[str, object]
    if dual is not None:
        replay_error = (
            (replay_peakness.to(dtype=dual.surface_peakness.dtype) - dual.surface_peakness)
            .abs()
            .max()
        )
        state_report = {
            "appearance_confidence": _tensor_summary(dual.appearance_confidence),
            "surface_peakness": _tensor_summary(dual.surface_peakness),
            "surface_localization_support": _tensor_summary(dual.surface_localization_support),
            "localization_completion_gate": _tensor_summary(dual.localization_completion_gate),
            "surface_peakness_replay_max_abs_error": float(replay_error.item()),
        }
    else:
        if transport is None:
            raise RuntimeError("transport evidence unexpectedly missing")
        counterfactuals = candidate_profiles.get("counterfactuals")
        if not isinstance(counterfactuals, dict) or "min_two_observers" not in counterfactuals:
            raise RuntimeError("transport telemetry requires min-two-observer profiles")
        candidate_profiles = counterfactuals["min_two_observers"]
        replay = _transport_surface_evidence(
            batch.context_rgb,
            batch.context_cameras,
            batch.bounds,
            config.model.voxel_resolution,
            transport.appearance_confidence,
            evidence_temperature=config.model.evidence_temperature,
            surface_peak_temperature=config.model.surface_peak_temperature,
            chunk_size=trainer.model.state_encoder.surface_peak_chunk_size,
        )
        observed = (
            transport.transport_coverage,
            transport.transported_appearance_confidence,
            transport.surface_peakness,
            transport.surface_localization_support,
        )
        replay_errors = [
            float((actual.to(dtype=expected.dtype) - expected).abs().max().item())
            for actual, expected in zip(replay, observed, strict=True)
        ]
        state_report = {
            "appearance_confidence": _tensor_summary(transport.appearance_confidence),
            "transported_appearance_confidence": _tensor_summary(
                transport.transported_appearance_confidence
            ),
            "surface_peakness": _tensor_summary(transport.surface_peakness),
            "surface_localization_support": _tensor_summary(transport.surface_localization_support),
            "transport_coverage": _tensor_summary(transport.transport_coverage),
            "localization_completion_gate": _tensor_summary(transport.localization_completion_gate),
            "transport_replay_max_abs_error": max(replay_errors),
            "transport_replay_field_max_abs_errors": {
                name: value
                for name, value in zip(
                    (
                        "transport_coverage",
                        "transported_appearance_confidence",
                        "surface_peakness",
                        "surface_localization_support",
                    ),
                    replay_errors,
                    strict=True,
                )
            },
        }
    gated_residual = evidence.localization_completion_gate * evidence.density_residual
    reconstructed_density = evidence.base_density_logits + gated_residual
    reconstruction_error = (reconstructed_density - output.state.density_logits).abs().max()
    state_report["density_decomposition"] = {
        "base_density_logits": _tensor_summary(evidence.base_density_logits),
        "gated_density_residual": _tensor_summary(gated_residual),
        "final_density_logits": _tensor_summary(output.state.density_logits),
        "reconstruction_max_abs_error": float(reconstruction_error.item()),
    }

    report: dict[str, object] = {
        "schema_version": "mcss.v6_evidence_telemetry.v1",
        "architecture": config.model.state_architecture,
        "diagnostic_only": True,
        "target_labels_used": False,
        "target_camera_role": "query_only",
        "checkpoint_selection_allowed": False,
        "source": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_step": trainer.step,
            "dataset_index": dataset_index,
            "scene_ids": list(batch.scene_ids),
        },
        "resolved_config": json.loads(json.dumps(asdict(config))),
        "candidate_profiles": candidate_profiles,
        "state": state_report,
    }
    _atomic_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit V6 depth-candidate evidence without reading target labels."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-index", type=int, default=0)
    args = parser.parse_args(argv)
    report = run_evidence_telemetry(
        args.config,
        args.checkpoint,
        args.output_dir,
        dataset_index=args.dataset_index,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
