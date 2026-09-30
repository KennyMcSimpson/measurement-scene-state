"""Read-only RGB provenance audit; no carrier predictions or frame exclusions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mcss.mechanism_pilot.data_prep import resize_rgb_depth
from mcss.mechanism_pilot.small_training import TrainScene


def rgb_statistics(array: np.ndarray, *, scale: float = 255.0) -> dict:
    """Record RGB pixel statistics before any preprocessing; near-black <= 1/255."""
    x = np.asarray(array)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("Expected HWC RGB")
    finite = np.isfinite(x)
    values = x[finite].astype(np.float64)
    pixel_finite = finite.all(-1)
    return {
        "dimensions_hw": list(x.shape[:2]),
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else None,
        "exact_zero_pixels_percent": float(((x == 0).all(-1) & pixel_finite).mean() * 100),
        "near_black_pixels_percent": float(
            ((x >= 0).all(-1) & (x <= scale / 255).all(-1) & pixel_finite).mean() * 100
        ),
        "near_black_definition": "all RGB channels in [0,1/255] of declared full range",
        "unique_rgb_values": int(np.unique(x.reshape(-1, 3), axis=0).shape[0]),
        "nan_values": int(np.isnan(x).sum()),
        "inf_values": int(np.isinf(x).sum()),
        "declared_full_range": scale,
    }


def file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def classify_black_frame(row: dict) -> str:
    """Localize a black prepared frame without guessing the dataset's render cause."""
    if not row["frame_mapping_valid"]:
        return "FRAME_MAPPING_ERROR"
    if row["decoder_black_agreement"] is False:
        return "DECODER_ERROR"
    tensor = row["model_input"]
    if (
        tensor["min"] is None
        or tensor["min"] < 0
        or tensor["max"] > 1
        or tensor["nan_values"]
        or tensor["inf_values"]
    ):
        return "RANGE_ERROR"
    if not row["prepared_matches_recomputed_resize"]:
        return "PREPROCESSING_ERROR"
    if row["source_rgb"]["max"] == 0 and row["prepared_rgb"]["max"] == 0:
        return "SOURCE_FRAME_IS_BLACK"
    return "UNKNOWN"


def _official_rows(directory: Path | None, scene: str, frame: int) -> dict:
    if directory is None:
        return {"status": "NOT_CHECKED"}
    result = {}
    for name in (
        "metadata_images.csv",
        "metadata_images_split_scene_v1.csv",
        "metadata_camera_trajectories.csv",
    ):
        path = directory / name
        if not path.is_file():
            result[name] = {"status": "MISSING"}
            continue
        with path.open() as handle:
            records = list(csv.DictReader(handle))
        if name == "metadata_camera_trajectories.csv":
            matches = [
                r
                for r in records
                if r.get("Animation") == f"{scene}_cam_00"
                or (r.get("scene_name") == scene and r.get("camera_name") == "cam_00")
            ]
        else:
            matches = [
                r
                for r in records
                if r.get("scene_name") == scene
                and r.get("camera_name") == "cam_00"
                and str(r.get("frame_id", "")).lstrip("0") == str(frame).lstrip("0")
            ]
        result[name] = {"matching_rows": matches}
    return result


def audit_rgb_chain(manifest_path, output_dir, official_metadata_dir=None) -> dict:
    """Audit existing source JPEGs and exact current preparation/loader outputs."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frame_audit.json").exists():
        raise FileExistsError("Keep previous audit; use a fresh output directory")
    metadata = Path(official_metadata_dir) if official_metadata_dir else None
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for record in manifest["scenes"]:
        scene = TrainScene(record, manifest["image_size"], manifest_path.parent, "cpu", [])
        for frame in record["frames"]:
            fid = frame["frame_id"]
            source = (
                manifest_path.parent
                / "raw"
                / record["scene_id"]
                / f"images/scene_cam_00_final_preview/frame.{fid:04d}.color.jpg"
            )
            prepared = scene.path(frame, "rgb")
            with Image.open(source) as image:
                image.load()
                native = np.array(image)
                rgb = np.array(image.convert("RGB"))
                source_mode, source_format = image.mode, image.format
            with Image.open(prepared) as image:
                saved = np.array(image.convert("RGB"))
            resized, _ = resize_rgb_depth(
                rgb, np.ones(rgb.shape[:2], np.float32), tuple(manifest["image_size"])
            )
            tensor = scene.rgb(frame)
            evaluator = scene.rgb(frame)
            decoder = {"status": "UNAVAILABLE"}
            try:
                import cv2

                other = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if other is None:
                    raise ValueError("Independent OpenCV decoder failed")
                other = other[..., ::-1]
                decoder = {
                    "status": "DECODED",
                    "rgb": rgb_statistics(other),
                    "exact_pixel_match": bool(np.array_equal(rgb, other)),
                }
            except ImportError:
                pass
            row = {
                "scene_id": record["scene_id"],
                "frame_id": fid,
                "source_file": file_record(source),
                "prepared_file": file_record(prepared),
                "source_format": source_format,
                "source_mode": source_mode,
                "native_array_shape": list(native.shape),
                "native_array_dtype": str(native.dtype),
                "source_rgb": rgb_statistics(rgb),
                "recomputed_resize_rgb": rgb_statistics(resized),
                "prepared_rgb": rgb_statistics(saved),
                "model_input": rgb_statistics(tensor.permute(1, 2, 0).numpy(), scale=1),
                "evaluator_input": rgb_statistics(evaluator.permute(1, 2, 0).numpy(), scale=1),
                "model_evaluator_tensors_equal": bool(torch.equal(tensor, evaluator)),
                "prepared_matches_recomputed_resize": bool(np.array_equal(saved, resized)),
                "independent_decoder": decoder,
                "decoder_black_agreement": None
                if decoder["status"] != "DECODED"
                else bool((decoder["rgb"]["max"] == 0) == (rgb.max() == 0)),
                "frame_mapping_valid": prepared.stem == f"{fid:04d}"
                and prepared.parent.name == record["scene_id"],
                "official_metadata": _official_rows(metadata, record["scene_id"], fid),
            }
            row["ROOT_CAUSE"] = classify_black_frame(row)
            rows.append(row)
    ai003 = [r for r in rows if r["scene_id"] == "ai_003_001"]
    black = [r for r in ai003 if r["prepared_rgb"]["max"] == 0]
    verdict = (
        "SOURCE_FRAME_IS_BLACK"
        if black and all(r["ROOT_CAUSE"] == "SOURCE_FRAME_IS_BLACK" for r in black)
        else "UNKNOWN"
    )
    root_cause = {
        "ROOT_CAUSE": verdict,
        "black_frames": [r["frame_id"] for r in black],
        "frames_audited": len(ai003),
        "render_failure_status": "UNKNOWN_NOT_INFERRED_FROM_BLACK_PIXELS",
        "frame_exclusion_applied": False,
        "RGB_PREPROCESSING_CHANGED": False,
        "scope": "Existing TRAIN media only, all frames retained; no model outputs used",
        "source_is": "Published final_preview JPEG; not the linear HDR color buffer",
        "official_metadata_directory": str(metadata) if metadata else None,
        "near_black_rule": "descriptive only, all channels <= 1/255; not an exclusion rule",
        "future_unseen_quality_rule": (
            "official included; camera not BAD; finite positive depth >= .95; "
            "reject all-black or >99.9% pixels all RGB channels <=2/255; no model scores"
        ),
        "future_rule_applied_to_existing_train": False,
    }
    for name, data in (("frame_audit.json", rows), ("ai003_black_root_cause.json", root_cause)):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    fields = [
        "scene_id",
        "frame_id",
        "stage",
        "path",
        "file_size",
        "sha256",
        "dimensions_hw",
        "dtype",
        "min",
        "max",
        "mean",
        "std",
        "exact_zero_pixels_percent",
        "near_black_pixels_percent",
        "unique_rgb_values",
        "nan_values",
        "inf_values",
    ]
    with (output / "ai003_frame_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in ai003:
            for stage in (
                "source_rgb",
                "recomputed_resize_rgb",
                "prepared_rgb",
                "model_input",
                "evaluator_input",
            ):
                file = r["source_file" if stage == "source_rgb" else "prepared_file"]
                writer.writerow(
                    {
                        "scene_id": r["scene_id"],
                        "frame_id": r["frame_id"],
                        "stage": stage,
                        "path": file["path"],
                        "file_size": file["bytes"],
                        "sha256": file["sha256"],
                        **{k: r[stage][k] for k in fields[6:]},
                    }
                )
    return root_cause
