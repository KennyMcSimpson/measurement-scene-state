"""Deterministic Hypersim expansion planning with a sealed final holdout."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

OFFICIAL_SPLITS = ("train", "val", "test")
PROTOCOL_PARTITIONS = ("train", "val", "diagnostic_test", "final_holdout")
_SCENE_NAME = re.compile(r"^ai_\d{3}_\d{3}$")
_CAMERA_NAME = re.compile(r"^cam_\d{2}$")


@dataclass(frozen=True)
class ExpansionScene:
    scene_name: str
    official_split: str
    protocol_partition: str
    selected_camera: str
    frame_count: int
    previously_observed: bool


@dataclass(frozen=True)
class ExpansionExclusion:
    scene_name: str
    selected_camera: str
    frame_count: int
    reason: str = "previously_evaluated_test_scene"


@dataclass(frozen=True)
class ExpansionPlan:
    scenes: tuple[ExpansionScene, ...]
    exclusions: tuple[ExpansionExclusion, ...]
    metadata_sha256: str

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(scene.protocol_partition for scene in self.scenes)
        return {partition: counts[partition] for partition in PROTOCOL_PARTITIONS}

    @property
    def official_counts(self) -> dict[str, int]:
        counts = Counter(scene.official_split for scene in self.scenes)
        return {split: counts[split] for split in OFFICIAL_SPLITS}

    @property
    def final_holdout_scenes(self) -> frozenset[str]:
        return frozenset(
            scene.scene_name
            for scene in self.scenes
            if scene.protocol_partition == "final_holdout"
        )


@dataclass(frozen=True)
class StorageEstimate:
    current_free_bytes: int
    required_free_bytes: int
    local_raw_scenes: int
    missing_raw_scenes: int
    raw_bytes_per_frame: float
    prepared_bytes_per_frame: float
    estimated_incremental_raw_bytes: int
    estimated_incremental_prepared_bytes: int
    safety_factor: float

    @property
    def estimated_incremental_total_bytes(self) -> int:
        return self.estimated_incremental_raw_bytes + self.estimated_incremental_prepared_bytes

    @property
    def estimated_free_after_bytes(self) -> int:
        return self.current_free_bytes - self.estimated_incremental_total_bytes

    @property
    def fits_budget(self) -> bool:
        return self.estimated_free_after_bytes >= self.required_free_bytes

    def as_dict(self) -> dict[str, int | float | bool]:
        result: dict[str, int | float | bool] = asdict(self)
        result["estimated_incremental_total_bytes"] = self.estimated_incremental_total_bytes
        result["estimated_free_after_bytes"] = self.estimated_free_after_bytes
        result["fits_budget"] = self.fits_budget
        return result


def build_expansion_plan(
    metadata_path: str | Path,
    observed_subset_path: str | Path,
) -> ExpansionPlan:
    metadata = Path(metadata_path)
    official_scenes = _load_official_scenes(metadata)
    observed_splits = _load_observed_subset(Path(observed_subset_path))

    unknown = sorted(set(observed_splits) - set(official_scenes))
    if unknown:
        raise ValueError(
            f"observed subset contains scenes absent from official metadata: {unknown}"
        )
    for scene_name, observed_split in observed_splits.items():
        official_split = official_scenes[scene_name][0]
        if observed_split != official_split:
            raise ValueError(
                f"split mismatch for {scene_name}: observed={observed_split}, "
                f"official={official_split}"
            )

    scenes: list[ExpansionScene] = []
    exclusions: list[ExpansionExclusion] = []
    for scene_name, (official_split, camera, frame_count) in sorted(official_scenes.items()):
        previously_observed = scene_name in observed_splits
        if official_split == "test":
            protocol_partition = (
                "diagnostic_test" if previously_observed else "final_holdout"
            )
        else:
            protocol_partition = official_split
        scene = ExpansionScene(
            scene_name=scene_name,
            official_split=official_split,
            protocol_partition=protocol_partition,
            selected_camera=camera,
            frame_count=frame_count,
            previously_observed=previously_observed,
        )
        scenes.append(scene)
        if protocol_partition == "diagnostic_test":
            exclusions.append(
                ExpansionExclusion(
                    scene_name=scene_name,
                    selected_camera=camera,
                    frame_count=frame_count,
                )
            )

    final_holdout = {
        scene.scene_name for scene in scenes if scene.protocol_partition == "final_holdout"
    }
    excluded = {row.scene_name for row in exclusions}
    if final_holdout & excluded:
        raise RuntimeError("final holdout intersects previously evaluated test scenes")
    return ExpansionPlan(tuple(scenes), tuple(exclusions), _sha256(metadata))


def write_expansion_plan(
    plan: ExpansionPlan,
    partitions_path: str | Path,
    exclusions_path: str | Path,
) -> None:
    _write_dataclass_csv(Path(partitions_path), plan.scenes)
    _write_dataclass_csv(Path(exclusions_path), plan.exclusions)


def select_materialization_scenes(
    plan: ExpansionPlan,
    partitions: tuple[str, ...] | list[str],
    *,
    allow_final_holdout: bool = False,
) -> tuple[ExpansionScene, ...]:
    requested = tuple(dict.fromkeys(partitions))
    if not requested:
        raise ValueError("at least one protocol partition is required")
    invalid = sorted(set(requested) - set(PROTOCOL_PARTITIONS))
    if invalid:
        raise ValueError(f"unsupported protocol partitions: {invalid}")
    if "final_holdout" in requested and not allow_final_holdout:
        raise ValueError("final_holdout is sealed until explicitly released")
    selected = tuple(scene for scene in plan.scenes if scene.protocol_partition in requested)
    if not selected:
        raise ValueError(f"no scenes found for protocol partitions: {requested}")
    return selected


def estimate_expansion_storage(
    plan: ExpansionPlan,
    *,
    raw_root: str | Path,
    baseline_prepared_root: str | Path,
    target_prepared_root: str | Path,
    reserve_bytes: int,
    safety_factor: float = 1.25,
) -> StorageEstimate:
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be non-negative")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be at least one")
    raw = Path(raw_root)
    baseline_prepared = Path(baseline_prepared_root)
    target_prepared = Path(target_prepared_root)
    by_name = {scene.scene_name: scene for scene in plan.scenes}

    local_raw = [scene for scene in plan.scenes if (raw / scene.scene_name).is_dir()]
    raw_sample_bytes = sum(_directory_size(raw / scene.scene_name) for scene in local_raw)
    raw_sample_frames = sum(scene.frame_count for scene in local_raw)
    raw_bytes_per_frame = _rate(raw_sample_bytes, raw_sample_frames, "raw Hypersim")
    missing_raw = [scene for scene in plan.scenes if not (raw / scene.scene_name).is_dir()]

    prepared_samples: list[tuple[Path, ExpansionScene]] = []
    for split in ("train", "val", "test"):
        split_root = baseline_prepared / split
        if not split_root.is_dir():
            continue
        for scene_path in split_root.iterdir():
            if scene_path.is_dir() and scene_path.name in by_name:
                prepared_samples.append((scene_path, by_name[scene_path.name]))
    prepared_sample_bytes = sum(_directory_size(path) for path, _ in prepared_samples)
    prepared_sample_frames = sum(scene.frame_count for _, scene in prepared_samples)
    prepared_bytes_per_frame = _rate(
        prepared_sample_bytes, prepared_sample_frames, "prepared Hypersim"
    )

    missing_prepared = [
        scene
        for scene in plan.scenes
        if not (target_prepared / scene.protocol_partition / scene.scene_name).is_dir()
    ]
    incremental_raw = int(
        sum(scene.frame_count for scene in missing_raw) * raw_bytes_per_frame * safety_factor
    )
    incremental_prepared = int(
        sum(scene.frame_count for scene in missing_prepared)
        * prepared_bytes_per_frame
        * safety_factor
    )
    current_free = shutil.disk_usage(raw.resolve().anchor).free
    return StorageEstimate(
        current_free_bytes=current_free,
        required_free_bytes=reserve_bytes,
        local_raw_scenes=len(local_raw),
        missing_raw_scenes=len(missing_raw),
        raw_bytes_per_frame=raw_bytes_per_frame,
        prepared_bytes_per_frame=prepared_bytes_per_frame,
        estimated_incremental_raw_bytes=incremental_raw,
        estimated_incremental_prepared_bytes=incremental_prepared,
        safety_factor=safety_factor,
    )


def _load_official_scenes(path: Path) -> dict[str, tuple[str, str, int]]:
    required = {
        "scene_name",
        "camera_name",
        "frame_id",
        "included_in_public_release",
        "split_partition_name",
    }
    scene_splits: dict[str, set[str]] = defaultdict(set)
    camera_frames: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames, required)
        for row in reader:
            if row["included_in_public_release"].strip().lower() != "true":
                continue
            scene = row["scene_name"].strip()
            camera = row["camera_name"].strip()
            split = row["split_partition_name"].strip()
            if not _SCENE_NAME.fullmatch(scene):
                raise ValueError(f"invalid official scene name: {scene!r}")
            if not _CAMERA_NAME.fullmatch(camera):
                raise ValueError(f"invalid official camera name: {camera!r}")
            if split not in OFFICIAL_SPLITS:
                raise ValueError(f"unsupported official split for {scene}: {split!r}")
            try:
                frame_id = int(row["frame_id"])
            except ValueError as error:
                raise ValueError(f"invalid frame id for {scene}/{camera}") from error
            if frame_id < 0:
                raise ValueError(f"negative frame id for {scene}/{camera}")
            scene_splits[scene].add(split)
            camera_frames[scene][camera].add(frame_id)

    scenes: dict[str, tuple[str, str, int]] = {}
    for scene in sorted(scene_splits):
        splits = scene_splits[scene]
        if len(splits) != 1:
            raise ValueError(f"official scene crosses splits: {scene} -> {sorted(splits)}")
        cameras = camera_frames[scene]
        selected_camera = "cam_00" if "cam_00" in cameras else min(
            cameras,
            key=lambda camera: (-len(cameras[camera]), camera),
        )
        scenes[scene] = (next(iter(splits)), selected_camera, len(cameras[selected_camera]))
    if not scenes:
        raise ValueError("official metadata contains no public scenes")
    return scenes


def _load_observed_subset(path: Path) -> dict[str, str]:
    required = {"scene_name", "split_partition_name"}
    scenes: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames, required)
        for row in reader:
            scene = row["scene_name"].strip()
            split = row["split_partition_name"].strip()
            if scene in scenes:
                raise ValueError(f"duplicate observed scene: {scene}")
            if split not in OFFICIAL_SPLITS:
                raise ValueError(f"unsupported observed split for {scene}: {split!r}")
            scenes[scene] = split
    return scenes


def _require_columns(path: Path, fieldnames: list[str] | None, required: set[str]) -> None:
    missing = required - set(fieldnames or ())
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def _write_dataclass_csv(path: Path, rows: tuple[object, ...]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    first = asdict(rows[0])  # type: ignore[arg-type]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))  # type: ignore[arg-type]
    temporary.replace(path)


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _rate(byte_count: int, frame_count: int, label: str) -> float:
    if byte_count <= 0 or frame_count <= 0:
        raise ValueError(f"cannot estimate {label} storage without a non-empty local sample")
    return byte_count / frame_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
