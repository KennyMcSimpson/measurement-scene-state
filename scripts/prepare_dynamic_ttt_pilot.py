"""Prepare a small, deterministic development manifest; does not run dynamic TTT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mcss.data.manifest import load_manifest, manifest_relative_path

PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_episode(path: Path, camera: str, split: str) -> dict:
    manifest = load_manifest(path)
    frames = sorted(manifest.frames, key=lambda frame: frame.frame_id)
    if len(frames) < 16:
        raise ValueError(f"Too few real frames: {manifest.scene_id}")
    frames = frames[:16]
    records = []
    for index, frame in enumerate(frames):
        pose = np.asarray(frame.c2w)
        rotation = pose[:3, :3]
        intrinsics = np.asarray(frame.intrinsics)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError(f"Non-orthogonal rotation: {manifest.scene_id}/{frame.frame_id}")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError("Camera rotation must be right-handed")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError("Invalid focal length")
        record = {"frame_id": frame.frame_id, "intrinsics": frame.intrinsics, "c2w": frame.c2w}
        for modality in ("rgb", "depth", "normal"):
            reference = getattr(frame, modality)
            if reference is None:
                raise ValueError(f"Missing {modality} reference")
            source = manifest_relative_path(path.parent, reference)
            if not source.is_file():
                raise FileNotFoundError(source)
            record[modality] = source.as_posix()
            # Query content remains unopened, even during this development preflight.
            if index < 12:
                if modality == "rgb":
                    with Image.open(source) as image:
                        image.load()
                        if (image.height, image.width) != manifest.image_size:
                            raise ValueError(f"RGB shape mismatch: {source}")
                else:
                    values = np.load(source, allow_pickle=False)
                    expected = manifest.image_size + ((3,) if modality == "normal" else ())
                    if values.shape != expected or not np.isfinite(values).all():
                        raise ValueError(f"Invalid {modality} array: {source}")
                    if modality == "depth" and (values < 0).any():
                        raise ValueError(f"Negative ray-distance depth: {source}")
        records.append(record)
    return {
        "scene_id": manifest.scene_id,
        "split": split,
        "camera": camera,
        "source_manifest": path.resolve().as_posix(),
        "source_manifest_sha256": sha256(path),
        "image_size": list(manifest.image_size),
        "warmup": records[:4],
        "stream": records[4:12],
        "query": records[12:],
    }


def build_plan(prepared_root: Path, partitions: Path, scenes_per_split: int = 3) -> dict:
    if scenes_per_split < 1:
        raise ValueError("scenes_per_split must be positive")
    with partitions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_scene = {}
    for row in rows:
        name = row["scene_name"]
        if name in by_scene:
            raise ValueError(f"Duplicate scene partition: {name}")
        by_scene[name] = row
    episodes = []
    all_scene_ids: set[str] = set()
    for source_split, pilot_split in (("train", "train"), ("val", "dev")):
        candidates = []
        for path in sorted((prepared_root / source_split).glob("*/manifest.json")):
            manifest = load_manifest(path)
            name = manifest.scene_id
            if name in all_scene_ids:
                raise ValueError(f"Scene appears across splits: {name}")
            all_scene_ids.add(name)
            row = by_scene.get(name)
            if row is None or row["protocol_partition"] != source_split:
                raise ValueError(f"Partition mismatch: {name}")
            # Selection depends only on scene identity and available frame IDs.
            if len(manifest.frames) >= 16:
                candidates.append((path, row["selected_camera"]))
        if len(candidates) < scenes_per_split:
            raise ValueError(f"Not enough eligible {source_split} scenes")
        episodes.extend(
            make_episode(path, camera, pilot_split)
            for path, camera in candidates[:scenes_per_split]
        )
    return {
        "schema": "mcss.dynamic_ttt.pilot_plan.v1",
        "status": "development_data_preflight_only",
        "selection": "First eligible scene IDs per split; first 16 sorted real frame IDs",
        "partitions_csv": partitions.resolve().as_posix(),
        "partitions_sha256": sha256(partitions),
        "prepared_root": prepared_root.resolve().as_posix(),
        "runtime_isolation_verified": False,
        "dynamic_ttt_implemented": False,
        "online_inputs": ["arrived_rgb", "arrived_camera", "past_state", "past_actions"],
        "offline_labels_only": ["depth", "normal"],
        "query_policy": (
            "Exclude query views/labels from online state, action selection and updates"
        ),
        "spatial_protocol_required": "context_local_metric",
        "local_bounds_m": [[-6.0, -4.0, 0.05], [6.0, 4.0, 12.05]],
        "anchor": "First warmup camera; never use full-scene depth-derived manifest.bounds",
        "depth_convention": "Metric Euclidean ray distance, not camera z-depth",
        "camera_calibration": (
            "Prepared K: default 60-degree FOV; official calibration audit pending"
        ),
        "diagnostic_test_used": False,
        "final_holdout_used": False,
        "array_checks": "Warmup/stream RGB and depth/normal only; query content not opened",
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=PROJECT / "data/hypersim_er_prepared")
    parser.add_argument(
        "--partitions", type=Path, default=PROJECT / "configs/hypersim_er_partitions.csv"
    )
    parser.add_argument("--scenes-per-split", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.relative_to(PROJECT / "outputs")  # Keep preparation artifacts in project outputs.
    manifest_path = output / "pilot_manifest.json"
    hash_path = output / "pilot_manifest.sha256"
    if manifest_path.exists() or hash_path.exists():
        parser.error("Pilot artifacts already exist; choose a new output directory")
    plan = build_plan(args.prepared_root, args.partitions, args.scenes_per_split)
    output.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with hash_path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{sha256(manifest_path)}  pilot_manifest.json\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "episodes": len(plan["episodes"]),
                "status": plan["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
