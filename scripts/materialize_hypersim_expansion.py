"""Resume-safe download and preparation of the planned Hypersim expansion."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from mcss.data.download import download_hypersim_members
from mcss.data.hypersim import prepare_hypersim
from mcss.data.hypersim_expansion import (
    PROTOCOL_PARTITIONS,
    ExpansionScene,
    build_expansion_plan,
    select_materialization_scenes,
)
from mcss.data.manifest import load_manifest

GIB = 1024**3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("download", "prepare", "all"), required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/hypersim_raw/official_metadata/metadata_images_split_scene_v1.csv"),
    )
    parser.add_argument(
        "--observed-subset", type=Path, default=Path("configs/hypersim_subset_v1.csv")
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/hypersim_raw"))
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("data/hypersim_er_prepared")
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=PROTOCOL_PARTITIONS,
        default=["train", "val", "diagnostic_test"],
    )
    parser.add_argument("--allow-final-holdout", action="store_true")
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reserve-gib", type=float, default=100.0)
    parser.add_argument(
        "--log", type=Path, default=Path("outputs/hypersim_er_materialization/materialize.jsonl")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.reserve_gib < 0:
        raise ValueError("--reserve-gib must be non-negative")
    if args.image_height < 1 or args.image_width < 1:
        raise ValueError("image dimensions must be positive")

    plan = build_expansion_plan(args.metadata, args.observed_subset)
    scenes = select_materialization_scenes(
        plan,
        args.partitions,
        allow_final_holdout=args.allow_final_holdout,
    )
    if args.limit is not None:
        scenes = scenes[: args.limit]
    summary = {
        "stage": args.stage,
        "dry_run": args.dry_run,
        "scene_count": len(scenes),
        "partitions": list(dict.fromkeys(scene.protocol_partition for scene in scenes)),
        "metadata_sha256": plan.metadata_sha256,
        "reserve_bytes": int(args.reserve_gib * GIB),
    }
    print(json.dumps({"event": "plan", **summary}, sort_keys=True), flush=True)
    if args.dry_run:
        for scene in scenes:
            print(json.dumps({"event": "scene", **asdict(scene)}, sort_keys=True))
        return 0

    args.log.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _materialize_one,
                scene,
                stage=args.stage,
                raw_root=args.raw_root,
                prepared_root=args.prepared_root,
                image_size=(args.image_height, args.image_width),
                reserve_bytes=int(args.reserve_gib * GIB),
            ): scene
            for scene in scenes
        }
        for future in as_completed(futures):
            scene = futures[future]
            try:
                record = future.result()
                completed += 1
            except Exception as error:  # Continue so a transient scene failure is resumable.
                record = {
                    "event": "failed",
                    "scene_name": scene.scene_name,
                    "protocol_partition": scene.protocol_partition,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(record)
            _append_jsonl(args.log, record)
            print(
                json.dumps(
                    {**record, "progress": f"{completed + len(failures)}/{len(scenes)}"},
                    sort_keys=True,
                ),
                flush=True,
            )

    final = {
        "event": "summary",
        **summary,
        "completed": completed,
        "failed": len(failures),
        "free_bytes": shutil.disk_usage(args.raw_root.resolve().anchor).free,
    }
    _append_jsonl(args.log, final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0 if not failures else 1


def _materialize_one(
    scene: ExpansionScene,
    *,
    stage: str,
    raw_root: Path,
    prepared_root: Path,
    image_size: tuple[int, int],
    reserve_bytes: int,
) -> dict[str, object]:
    started = time.perf_counter()
    downloaded = False
    prepared = False
    _require_space(raw_root, reserve_bytes)
    if stage in {"download", "all"} and not _raw_scene_complete(raw_root, scene):
        download_hypersim_members(
            scene.scene_name,
            raw_root,
            camera=scene.selected_camera,
            frame_ids=None,
        )
        downloaded = True
    if stage in {"prepare", "all"}:
        if not _raw_scene_complete(raw_root, scene):
            raise FileNotFoundError(
                f"raw scene is incomplete: {scene.scene_name}/{scene.selected_camera}"
            )
        destination = prepared_root / scene.protocol_partition / scene.scene_name
        if not _prepared_scene_complete(destination, scene.frame_count):
            _require_space(prepared_root, reserve_bytes)
            paths = prepare_hypersim(
                raw_root,
                prepared_root / scene.protocol_partition,
                scenes=[scene.scene_name],
                camera=scene.selected_camera,
                image_size=image_size,
                frame_stride=1,
            )
            if len(paths) != 1:
                raise RuntimeError(f"expected one prepared manifest for {scene.scene_name}")
            prepared = True
    _require_space(raw_root, reserve_bytes)
    return {
        "event": "completed",
        "scene_name": scene.scene_name,
        "protocol_partition": scene.protocol_partition,
        "selected_camera": scene.selected_camera,
        "frame_count": scene.frame_count,
        "downloaded": downloaded,
        "prepared": prepared,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _raw_scene_complete(root: Path, scene: ExpansionScene) -> bool:
    scene_root = root / scene.scene_name
    detail = scene_root / "_detail"
    camera_detail = detail / scene.selected_camera
    preview = scene_root / "images" / f"scene_{scene.selected_camera}_final_preview"
    geometry = scene_root / "images" / f"scene_{scene.selected_camera}_geometry_hdf5"
    required = (
        detail / "metadata_scene.csv",
        detail / "metadata_cameras.csv",
        camera_detail / "camera_keyframe_frame_indices.hdf5",
        camera_detail / "camera_keyframe_look_at_positions.hdf5",
        camera_detail / "camera_keyframe_orientations.hdf5",
        camera_detail / "camera_keyframe_positions.hdf5",
    )
    return (
        all(path.is_file() and path.stat().st_size > 0 for path in required)
        and len(list(preview.glob("frame.*.color.jpg"))) == scene.frame_count
        and len(list(geometry.glob("frame.*.depth_meters.hdf5"))) == scene.frame_count
        and len(list(geometry.glob("frame.*.normal_world.hdf5"))) == scene.frame_count
    )


def _prepared_scene_complete(destination: Path, frame_count: int) -> bool:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        return len(load_manifest(manifest_path).frames) == frame_count
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _require_space(path: Path, reserve_bytes: int) -> None:
    free = shutil.disk_usage(path.resolve().anchor).free
    if free < reserve_bytes:
        raise OSError(f"free-space reserve violated: {free} < {reserve_bytes} bytes")


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
