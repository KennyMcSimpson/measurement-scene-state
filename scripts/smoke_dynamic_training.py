"""Run a finite, train-only dynamic training smoke on ai_001_001."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcss.data.episodes import OnlineEpisodeSource
from mcss.data.pilot_provenance import validate_pilot_entries, write_pilot_provenance
from mcss.training.smoke import DynamicTrainingSmokeConfig, run_dynamic_training_smoke
from mcss.training.supervision import TrainingSupervision

PROJECT = Path(__file__).resolve().parents[1]
EPISODE_INDEX_SCHEMA = "mcss.dynamic.episode_index.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-index",
        type=Path,
        default=PROJECT / "outputs/dynamic_ttt_pilot_20260919/episodes/episode_index.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/dynamic_ttt_pilot_20260919/training_smoke_verified",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--renderer-samples", type=int, default=8)
    parser.add_argument("--ray-chunk-size", type=int, default=2048)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.relative_to(PROJECT / "outputs")
    if (output_dir / "report.json").exists():
        raise FileExistsError(f"refusing to overwrite existing smoke report: {output_dir}")
    device = _resolve_device(args.device)
    torch.manual_seed(20260919)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260919)

    entry = _first_train_entry(args.episode_index)
    provenance = validate_pilot_entries((entry,))
    source = OnlineEpisodeSource.from_file(entry["online_manifest"], device=device)
    _validate_online_source(source, entry)
    warmup = source.warmup()
    stream = tuple(source.next_observation() for _ in range(2))
    context_frame_ids = tuple(observation.frame_id for observation in (*warmup, *stream))
    supervision = TrainingSupervision.from_query_file(
        entry["query_manifest"],
        episode_id=source.episode_id,
        scene_id=source.scene_id,
        query_vault_id=source.query_vault_id,
        context_frame_ids=context_frame_ids,
        device=device,
    )
    config = DynamicTrainingSmokeConfig(
        renderer_samples=args.renderer_samples,
        ray_chunk_size=args.ray_chunk_size,
    )
    try:
        report = run_dynamic_training_smoke(
            warmup,
            stream,
            supervision,
            output_dir,
            device=device,
            config=config,
        )
    except torch.OutOfMemoryError:
        if device.type != "cuda" or args.renderer_samples <= 2:
            raise
        torch.cuda.empty_cache()
        reduced = DynamicTrainingSmokeConfig(
            renderer_samples=max(2, args.renderer_samples // 2),
            ray_chunk_size=args.ray_chunk_size,
        )
        report = run_dynamic_training_smoke(
            warmup,
            stream,
            supervision,
            output_dir,
            device=device,
            config=reduced,
            resource_adaptation=(
                f"CUDA OOM at renderer_samples={args.renderer_samples}; retried native "
                f"128x160 frames with renderer_samples={reduced.renderer_samples}"
            ),
    )
    provenance_path = write_pilot_provenance(output_dir / "manifest_provenance.json", provenance)
    result = {**report, "manifest_provenance": str(provenance_path)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _first_train_entry(index_path: Path) -> dict[str, object]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("schema_version") != EPISODE_INDEX_SCHEMA:
        raise ValueError("unexpected dynamic episode index schema")
    episodes = index.get("episodes")
    if not isinstance(episodes, list) or not episodes or not isinstance(episodes[0], dict):
        raise ValueError("dynamic episode index must contain an episode list")
    entry = episodes[0]
    expected = {
        "episode_id": "train--ai_001_001--cam_00",
        "scene_id": "ai_001_001",
        "split_id": "train",
        "stream_steps": 8,
        "query_count": 4,
    }
    if any(entry.get(name) != value for name, value in expected.items()):
        raise ValueError("first index entry is not the approved ai_001_001 train episode")
    return entry


def _validate_online_source(source: OnlineEpisodeSource, entry: dict[str, object]) -> None:
    if source.split_id != "train" or source.scene_id != "ai_001_001":
        raise ValueError("training smoke may only read the approved ai_001_001 train source")
    if source.episode_id != entry["episode_id"] or source.query_vault_id != entry["query_vault_id"]:
        raise ValueError("online source identity does not match the selected index entry")
    if source.remaining_steps < 2:
        raise ValueError("training smoke requires two available stream observations")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
