"""Metadata-only preparation report; never downloads media or launches training."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mcss.mechanism_pilot.calibration import calibrated_intrinsics, published_scene_metadata
from mcss.training.experiment import DynamicExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
NAME = "EXP-3D-20260927-training-preparation-v1"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "readiness.json").exists():
        raise FileExistsError("Use a new preparation output directory")
    metadata = ROOT / "outputs" / NAME / "metadata/metadata_camera_parameters.csv"
    published = published_scene_metadata(metadata)
    partitions_path = ROOT / "configs/hypersim_er_partitions.csv"
    with partitions_path.open() as f:
        partitions = list(csv.DictReader(f))
    candidates, used_volumes = [], set()
    for split in ["train", "val"]:
        chosen = []
        for row in sorted(partitions, key=lambda x: x["scene_name"]):
            scene = row["scene_name"]
            volume = scene.split("_")[1]
            if row["protocol_partition"] != split or int(row["frame_count"]) < 16:
                continue
            if volume in used_volumes or scene not in published:
                continue
            used_volumes.add(volume)
            k = calibrated_intrinsics(published[scene], (128, 160))
            chosen.append(
                {
                    "scene_id": scene,
                    "split": "dev" if split == "val" else "train",
                    "protocol_partition": split,
                    "camera": row["selected_camera"],
                    "selection": "first eligible scene in distinct asset volume; metadata only",
                    "physical_scene_id": None,
                    "physical_scene_identity_status": "UNVERIFIED",
                    "conservative_volume_group": volume,
                    "intrinsics": k.tolist(),
                    "frame_selection": "first 16 published available frames; exact IDs NOT LOCKED",
                    "prepared_manifest": None,
                    "media_present": False,
                }
            )
            if len(chosen) == 3:
                break
        candidates.extend(chosen)
    write(
        output / "cohort_draft.json",
        {
            "status": "DRAFT_NOT_AUTHORIZED_FOR_EVALUATION",
            "scope": (
                "small 3 train + 3 dev engineering/development pilot, not independent confirmation"
            ),
            "scene_selection_uses_scores": False,
            "records": candidates,
            "physical_identity_note": (
                "Different volume IDs are a conservative heuristic, not proof."
            ),
            "protected_partitions_selected": False,
        },
    )
    checks = []
    for scene, row in published.items():
        k = calibrated_intrinsics(row, (128, 160))
        checks.append(
            {
                "scene": scene,
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
                "skew_or_projective": not np.allclose(k[2], [0, 0, 1], atol=1e-9),
            }
        )
    write(
        output / "calibration_audit.json",
        {
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "rows_checked": len(checks),
            "calibration": "published M_cam_from_uv; half pixel; GL to CV",
            "image_size": [128, 160],
            "actual_media_geometry_check": "PENDING_NO_DATA",
            "rows": checks,
        },
    )
    base = DynamicExperimentConfig(seed=20260927, renderer_samples=64, stage_b_steps=0)
    write(output / "stage_a_config.json", asdict(base))
    write(
        output / "stage_b_config.json",
        asdict(
            DynamicExperimentConfig(
                seed=20260927, renderer_samples=64, stage_a_steps=0, stage_b_steps=600
            )
        ),
    )
    write(
        output / "training_protocol.json",
        {
            "status": "DRAFT_NOT_LAUNCHABLE",
            "stage_A_context_mode": "common_anchor_equal_contexts",
            "context_allocation": (
                "must be frozen with actual frame IDs; not legacy odd/even anchors"
            ),
            "stage_A_steps": 1000,
            "stage_B_steps": 600,
            "stage_B_gate": "static qualification reviewed first",
            "steps_are": "engineering starting budget, not convergence evidence",
            "data_image_size": [128, 160],
            "renderer_samples": 64,
            "renderer_note": (
                "align new mechanism pilot; explicit difference from historical 16 samples"
            ),
            "depth": "metric normalized-ray distance",
            "query": "offline train labels only; dev sealed",
            "bounds": "fixed context-local metric; never infer from query depth",
            "checkpoint_schema": "mcss.dynamic.v1",
            "old_trainer_compatible_json_is_not_launch_approval": True,
        },
    )
    blockers = [
        "NO_PREPARED_RGB_CAMERA_DEPTH_COHORT",
        "PHYSICAL_SCENE_IDENTITIES_UNVERIFIED",
        "FRAME_IDS_NOT_LOCKED",
        "CALIBRATED_PREPROCESSOR_NOT_INTEGRATED",
        "COMMON_ANCHOR_ADAPTER_NOT_INTEGRATED_IN_TRAINER",
        "RESIDUAL_CONTROL_TRAINING_PROTOCOL_PENDING",
    ]
    write(
        output / "readiness.json",
        {
            "TRAINING_READY": False,
            "status": "PREPARATION_COMPLETE_DATA_AND_INTEGRATION_PENDING",
            "blockers": blockers,
            "formal_training_steps": 0,
            "query_media_reads": 0,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "download_mode": (
                "bounded explicit member list only after physical identity/frame audit"
            ),
            "do_not_run": [
                "materialize_hypersim_expansion.py defaults include diagnostic_test",
                "legacy converter defaults assume 60 degree FOV and missing scale=1",
                "old trainer default phase A uses different anchors",
            ],
        },
    )
    print((output / "readiness.json").read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / NAME / "prepared")
    main(p.parse_args().output)
