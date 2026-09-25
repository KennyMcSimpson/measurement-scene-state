"""Command-line entry points for training, evaluation, data, and diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from mcss.config import ExperimentConfig, load_config
from mcss.data.download import (
    build_replica_plan,
    download_hypersim_members,
    execute_plan,
    extract_replica_parts,
)
from mcss.data.hypersim import prepare_hypersim
from mcss.data.replica import prepare_replica
from mcss.diagnostics import no_bypass_audit
from mcss.engine import Trainer


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        trainer = Trainer(load_config(args.config))
        checkpoint = trainer.train()
        print(json.dumps({"checkpoint": str(checkpoint), "step": trainer.step}))
        return 0
    if args.command == "evaluate":
        config = _with_evaluation_overrides(
            load_config(args.config),
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            ray_chunk_size=args.ray_chunk_size,
        )
        trainer = Trainer(config)
        metrics = trainer.evaluate(args.checkpoint)
        print(json.dumps(metrics, sort_keys=True))
        return 0
    if args.command == "diagnose":
        trainer = Trainer(load_config(args.config))
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint, load_optimizer=False)
        report = no_bypass_audit(trainer.model)
        payload = {
            "passed": report.passed,
            "renderer_parameter_count": report.renderer_parameter_count,
            "forbidden_forward_arguments": list(report.forbidden_forward_arguments),
            "reasons": list(report.reasons),
        }
        trainer.output_dir.mkdir(parents=True, exist_ok=True)
        (trainer.output_dir / "diagnostic.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "download":
        return _download_command(args)
    if args.command == "prepare":
        return _prepare_command(args)
    raise RuntimeError(f"unsupported command {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcss")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "diagnose"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name in {"evaluate", "diagnose"}:
            command.add_argument("--checkpoint", type=Path)
        if name == "evaluate":
            command.add_argument("--output-dir", type=Path)
            command.add_argument("--n-samples", type=int)
            command.add_argument("--ray-chunk-size", type=int)

    download = subparsers.add_parser("download")
    download.add_argument("dataset", choices=("replica", "hypersim"))
    download.add_argument("--root", required=True, type=Path)
    download.add_argument("--scenes", nargs="*", default=[])
    download.add_argument("--camera", default="cam_00")
    download.add_argument("--frames", nargs="*", type=int)
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--extract-to", type=Path)
    download.add_argument("--remove-parts", action="store_true")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("dataset", choices=("replica", "hypersim"))
    prepare.add_argument("--raw-root", required=True, type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--scenes", nargs="*")
    prepare.add_argument("--camera", default="cam_00")
    prepare.add_argument("--image-height", type=int)
    prepare.add_argument("--image-width", type=int)
    prepare.add_argument("--frame-limit", type=int)
    prepare.add_argument("--frame-stride", type=int, default=1)
    prepare.add_argument("--depth-scale", type=float)
    return parser


def _download_command(args: argparse.Namespace) -> int:
    if args.dataset == "replica":
        plan = build_replica_plan(args.root)
        if args.dry_run:
            print(
                json.dumps(
                    {"dataset": plan.dataset, "items": [item.url for item in plan.items]}, indent=2
                )
            )
            return 0
        results = execute_plan(plan)
        if args.extract_to:
            extract_replica_parts(plan, args.extract_to, remove_parts=args.remove_parts)
        print(json.dumps({"dataset": "replica", "files": len(results)}))
        return 0
    if not args.scenes:
        raise ValueError("Hypersim download requires --scenes ai_DDD_DDD [...]")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": "hypersim",
                    "scenes": args.scenes,
                    "camera": args.camera,
                    "frames": args.frames,
                },
                indent=2,
            )
        )
        return 0
    frame_ids = None if args.frames is None else set(args.frames)
    files = []
    for scene in args.scenes:
        files.extend(
            download_hypersim_members(scene, args.root, camera=args.camera, frame_ids=frame_ids)
        )
    print(json.dumps({"dataset": "hypersim", "files": len(files)}))
    return 0


def _prepare_command(args: argparse.Namespace) -> int:
    image_size = _optional_image_size(args.image_height, args.image_width)
    if args.dataset == "replica":
        path = prepare_replica(
            args.raw_root,
            args.output_root,
            frame_limit=args.frame_limit,
            image_size=image_size,
            depth_scale=args.depth_scale,
        )
        print(json.dumps({"manifest": str(path)}))
        return 0
    paths = prepare_hypersim(
        args.raw_root,
        args.output_root,
        scenes=args.scenes,
        camera=args.camera,
        image_size=image_size,
        frame_stride=args.frame_stride,
        frame_limit=args.frame_limit,
    )
    print(json.dumps({"manifests": [str(path) for path in paths]}))
    return 0


def _optional_image_size(height: int | None, width: int | None) -> tuple[int, int] | None:
    if height is None and width is None:
        return None
    if height is None or width is None or height < 2 or width < 2:
        raise ValueError("--image-height and --image-width must both be at least two")
    return height, width


def _with_evaluation_overrides(
    config: ExperimentConfig,
    *,
    output_dir: Path | None,
    n_samples: int | None,
    ray_chunk_size: int | None,
) -> ExperimentConfig:
    if n_samples is not None and n_samples < 2:
        raise ValueError("--n-samples must be at least two")
    if ray_chunk_size is not None and ray_chunk_size < 1:
        raise ValueError("--ray-chunk-size must be positive")
    model = replace(
        config.model,
        n_samples=config.model.n_samples if n_samples is None else n_samples,
        ray_chunk_size=(
            config.model.ray_chunk_size if ray_chunk_size is None else ray_chunk_size
        ),
    )
    if output_dir is None:
        return replace(config, model=model)

    resolved = output_dir.resolve()
    for name in ("evaluation.json", "evaluation_report.json"):
        report_path = resolved / name
        if report_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing evaluation report: {report_path}"
            )
    return replace(
        config,
        model=model,
        training=replace(config.training, output_dir=str(resolved)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
