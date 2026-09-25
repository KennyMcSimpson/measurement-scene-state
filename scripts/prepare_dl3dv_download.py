"""Inventory and selectively download the pinned DL3DV-140 benchmark.

The default command only writes an authenticated, revision-pinned inventory and
does not download images.  Add ``--download-metadata`` to materialize one
canonical ``nerfstudio/transforms.json`` per scene.  Add ``--images-4`` and
``--download-images`` only after the protocol owner has supplied the exact frame
selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcss.data.dl3dv_download import (
    DEFAULT_REPRESENTATION,
    build_inventory,
    download_selected,
    protocol_frame_selection,
    select_files,
    write_inventory_artifacts,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "outputs/dl3dv_download_20260920"
DEFAULT_DATA_ROOT = PROJECT / "data/dl3dv_benchmark"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        help="reuse an existing pinned inventory instead of querying the Hub tree",
    )
    parser.add_argument("--scenes", nargs="*", help="64-hex scene hashes; default is all 140")
    parser.add_argument("--representation", default=DEFAULT_REPRESENTATION)
    parser.add_argument("--include-cameras", action="store_true")
    parser.add_argument(
        "--images-4", action="store_true", help="include level-4 PNGs in the selection"
    )
    parser.add_argument("--frames", nargs="*", type=int)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--target-every", type=int, default=8)
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        help="download selected transforms.json/cameras.bin files",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="download selected images_4 files; requires --images-4",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.download_images and not args.images_4:
        raise ValueError("--download-images requires --images-4")
    if args.frames is not None and not args.images_4:
        raise ValueError("--frames requires --images-4")
    if args.protocol_json is not None and not args.images_4:
        raise ValueError("--protocol-json requires --images-4")
    if args.reserve_gib < 0:
        raise ValueError("--reserve-gib must be non-negative")
    if args.inventory_dir is None:
        inventory = build_inventory(workers=args.workers)
        artifacts = write_inventory_artifacts(
            inventory,
            args.output_dir,
            representation=args.representation,
        )
    else:
        from mcss.data.dl3dv_download import load_inventory_artifacts

        inventory = load_inventory_artifacts(args.inventory_dir)
        artifacts = {
            "metadata": args.inventory_dir / "benchmark-meta.csv",
            "tree": args.inventory_dir / "tree_inventory.jsonl",
            "manifest": args.inventory_dir / "manifest.json",
        }
    protocol_frames = None
    if args.protocol_json is not None:
        protocol_frames = protocol_frame_selection(
            args.protocol_json,
            inventory,
            data_root=args.data_root,
            representation=args.representation,
            target_every=args.target_every,
        )
    selected = select_files(
        inventory,
        args.data_root,
        scenes=args.scenes,
        representation=args.representation,
        include_images4=args.images_4,
        frame_ids=args.frames,
        frame_ids_by_scene=protocol_frames,
        include_cameras=args.include_cameras,
    )
    metadata_selected = tuple(
        item for item in selected if item.remote.kind in {"transforms", "cameras"}
    )
    image_selected = tuple(item for item in selected if item.remote.kind == "images_4")
    downloaded = None
    if args.download_metadata or args.download_images:
        requested = []
        if args.download_metadata:
            requested.extend(metadata_selected)
        if args.download_images:
            requested.extend(image_selected)
        downloaded = download_selected(
            requested,
            status_path=args.output_dir / "status.json",
            lock_path=args.output_dir / "download.lock",
            reserve_bytes=int(args.reserve_gib * 1024**3),
            max_attempts=args.max_attempts,
            workers=args.workers,
        )
    report = {
        "schema": "mcss.dl3dv.prepare.v1",
        "inventory": {key: str(value) for key, value in artifacts.items()},
        "revision": inventory.manifest(args.representation)["revision"],
        "scene_count": len(inventory.scenes),
        "selected_files": len(selected),
        "selected_metadata_files": len(metadata_selected),
        "selected_images_4_files": len(image_selected),
        "selected_bytes": sum(item.remote.size for item in selected),
        "downloaded": downloaded,
        "images_download_started": bool(args.download_images),
    }
    report_path = args.output_dir / "prepare_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
