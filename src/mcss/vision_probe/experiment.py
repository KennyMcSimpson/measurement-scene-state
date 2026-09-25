"""Run the exploratory, sequence-disjoint 2D observation probe.

The runner supports the prepared DAVIS fallback and the original HPatches
layout. It never uses masks or homographies while proposing an adaptation
action; those files are opened only by the evaluation functions.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mcss.vision_probe.adaptation import (
    ACTIONS,
    FastState,
    fit_readout,
    masks,
    materialize,
    probe_mse,
    proposals,
)
from mcss.vision_probe.geometry import mask_propagation_metrics

PROJECT = Path(__file__).resolve().parents[3]
SCHEMA = "mcss.vision_probe.exploratory.v1"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".webp")


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    kind: str
    root: Path
    frames: tuple[Path, ...]
    masks: tuple[Path | None, ...]
    source_split: str
    homographies: tuple[Path | None, ...] = ()

    def probe_frames(self) -> tuple[Path, Path, Path]:
        if len(self.frames) < 3:
            raise ValueError(f"sequence has fewer than three frames: {self.name}")
        return (
            self.frames[0],
            self.frames[len(self.frames) // 3],
            self.frames[(2 * len(self.frames)) // 3],
        )

    def probe_masks(self) -> tuple[Path | None, Path | None, Path | None]:
        if not self.masks:
            return (None, None, None)
        indices = (0, len(self.masks) // 3, (2 * len(self.masks)) // 3)
        return tuple(self.masks[index] for index in indices)  # type: ignore[return-value]

    def manifest_row(self, split: str) -> dict[str, Any]:
        return {
            "sequence": self.name,
            "kind": self.kind,
            "source_split": self.source_split,
            "assigned_split": split,
            "frame_count": len(self.frames),
            "frame_names": [path.name for path in self.frames],
            "mask_available": bool(self.masks),
        }


@dataclass
class FrameFeatures:
    features: torch.Tensor
    rgb: torch.Tensor
    original_size: Any
    image_size: Any
    input_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _find_frame(directory: Path, number: int) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{number}{extension}"
        if candidate.is_file():
            return candidate
    matches = sorted(
        path for path in directory.iterdir() if path.is_file() and path.stem == str(number)
    )
    return matches[0] if matches else None


def _find_homography(directory: Path, target: int) -> Path | None:
    for name in (f"H_1_{target}", f"H_1_{target}.txt", f"H_1_{target}.homography"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _discover_davis(root: Path) -> tuple[SequenceInfo, ...] | None:
    dataset = root / "DAVIS" if (root / "DAVIS").is_dir() else root
    image_root = dataset / "JPEGImages" / "480p"
    mask_root = dataset / "Annotations" / "480p"
    split_root = dataset / "ImageSets" / "2017"
    if not image_root.is_dir() or not (split_root / "train.txt").is_file():
        return None
    names_by_split: dict[str, list[str]] = {}
    for split in ("train", "val"):
        path = split_root / f"{split}.txt"
        if path.is_file():
            names_by_split[split] = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    records: list[SequenceInfo] = []
    for source_split, names in names_by_split.items():
        for name in names:
            image_dir = image_root / name
            frames = tuple(
                sorted(
                    path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
            if len(frames) < 3:
                raise ValueError(f"DAVIS sequence has fewer than three frames: {name}")
            mask_dir = mask_root / name
            masks_for_frames = tuple(
                (mask_dir / f"{frame.stem}.png")
                if (mask_dir / f"{frame.stem}.png").is_file()
                else None
                for frame in frames
            )
            records.append(
                SequenceInfo(name, "davis", image_dir, frames, masks_for_frames, source_split)
            )
    return tuple(records)


def discover_sequences(data_root: str | Path) -> tuple[SequenceInfo, ...]:
    """Discover sequence metadata without opening images, masks, or geometry."""

    root = Path(data_root).resolve()
    davis = _discover_davis(root)
    if davis is not None:
        return davis
    candidates = [
        path for path in root.iterdir() if path.is_dir() and path.name[:1].lower() in {"i", "v"}
    ]
    if not candidates:
        candidates = sorted(
            {
                path.parent
                for path in root.rglob("1.*")
                if path.parent.name[:1].lower() in {"i", "v"}
            }
        )
    records: list[SequenceInfo] = []
    for directory in sorted(candidates, key=lambda path: _stable_rank(path.name)):
        frames = tuple(
            path for number in range(1, 4) if (path := _find_frame(directory, number)) is not None
        )
        if len(frames) != 3:
            raise ValueError(f"HPatches sequence is missing one of images 1..3: {directory.name}")
        homographies = (_find_homography(directory, 2), _find_homography(directory, 3))
        records.append(
            SequenceInfo(
                directory.name,
                directory.name[:1].lower(),
                directory,
                frames,
                (None, None, None),
                "hpatches",
                homographies,
            )
        )
    if not records:
        raise ValueError(f"no DAVIS or HPatches sequences found under {root}")
    return tuple(records)


def _proportional_quota(total: int, available: dict[str, int]) -> dict[str, int]:
    if not available or total == 0:
        return {key: 0 for key in available}
    denominator = sum(available.values())
    exact = {key: total * value / denominator for key, value in available.items()}
    quota = {key: min(available[key], int(value)) for key, value in exact.items()}
    for key in sorted(available, key=lambda item: (-(exact[item] - quota[item]), item)):
        if sum(quota.values()) >= total:
            break
        if quota[key] < available[key]:
            quota[key] += 1
    return quota


def split_sequences(
    sequences: Iterable[SequenceInfo],
    fit_count: int = 24,
    discovery_count: int = 12,
    validation_count: int = 16,
) -> dict[str, tuple[SequenceInfo, ...]]:
    """Assign fixed identity splits; unselected sequences remain sealed."""

    all_sequences = tuple(sequences)
    train = [sequence for sequence in all_sequences if sequence.source_split != "val"]
    sealed_official = [sequence for sequence in all_sequences if sequence.source_split == "val"]
    requested = fit_count + discovery_count + validation_count
    if min(fit_count, discovery_count, validation_count) < 0 or requested > len(train):
        raise ValueError(
            f"requested split counts exceed available train sequences: {requested}/{len(train)}"
        )
    remaining: dict[str, list[SequenceInfo]] = {}
    for sequence in train:
        remaining.setdefault(sequence.kind, []).append(sequence)
    for values in remaining.values():
        values.sort(key=lambda sequence: _stable_rank(sequence.name))
    result: dict[str, tuple[SequenceInfo, ...]] = {}
    for split, count in (
        ("fit", fit_count),
        ("discovery", discovery_count),
        ("validation", validation_count),
    ):
        quota = (
            _proportional_quota(count, {key: len(value) for key, value in remaining.items()})
            if any(key in {"i", "v"} for key in remaining)
            else {"davis": count}
        )
        if "davis" in remaining:
            quota = {"davis": count}
        selected: list[SequenceInfo] = []
        for kind in sorted(remaining):
            take = quota.get(kind, 0)
            selected.extend(remaining[kind][:take])
            del remaining[kind][:take]
        selected.sort(key=lambda sequence: _stable_rank(sequence.name))
        result[split] = tuple(selected)
    sealed = [sequence for values in remaining.values() for sequence in values]
    sealed.extend(sealed_official)
    result["sealed"] = tuple(sorted(sealed, key=lambda sequence: _stable_rank(sequence.name)))
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_status(output: Path, status: str, phase: str, **extra: Any) -> None:
    _write_json(
        output / "status.json", {"schema": SCHEMA, "status": status, "phase": phase, **extra}
    )


def _module_hash() -> str:
    try:
        import mcss.vision_probe.features as features_module

        source = Path(str(features_module.__file__)).resolve()
        return (
            _sha256_file(source)
            if source.is_file()
            else hashlib.sha256(str(source).encode()).hexdigest()
        )
    except (ImportError, OSError):
        return hashlib.sha256(b"features-module-unavailable").hexdigest()


def _normalise_extraction(value: Any) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
    if isinstance(value, dict):
        values = (value["features"], value["rgb"], value["original_size"], value["image_size"])
    elif hasattr(value, "features"):
        values = (value.features, value.rgb, value.original_size, value.image_size)
    else:
        values = tuple(value)
    features, rgb, original_size, image_size = values
    return (
        torch.as_tensor(features).detach().cpu().float(),
        torch.as_tensor(rgb).detach().cpu().float(),
        original_size,
        image_size,
    )


def _cached_extract(
    extractor: Any, image: Path, cache_root: Path, weight_hash: str, source_hash: str, size: int
) -> FrameFeatures:
    input_hash = _sha256_file(image)
    identity = {
        "input_sha256": input_hash,
        "weight_sha256": weight_hash,
        "model_source_sha256": source_hash,
        "size": size,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    tensor_path, metadata_path = cache_root / f"{key}.pt", cache_root / f"{key}.json"
    cache_root.mkdir(parents=True, exist_ok=True)
    if tensor_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == identity:
                try:
                    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(tensor_path, map_location="cpu")
                return FrameFeatures(
                    payload["features"],
                    payload["rgb"],
                    payload["original_size"],
                    payload["image_size"],
                    input_hash,
                )
        except (OSError, json.JSONDecodeError, KeyError, RuntimeError):
            pass
    features, rgb, original_size, image_size = _normalise_extraction(extractor.extract(image))
    torch.save(
        {
            "features": features,
            "rgb": rgb,
            "original_size": original_size,
            "image_size": image_size,
        },
        tensor_path,
    )
    _write_json(metadata_path, identity)
    return FrameFeatures(features, rgb, original_size, image_size, input_hash)


def _mask_grid(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        # DAVIS masks are palette images whose pixel values are object IDs.  Keep
        # those indices intact while resizing; converting to L would remap them.
        resized = image.resize((width, height), Image.Resampling.NEAREST)
        labels = np.asarray(resized, dtype=np.int64)
        if labels.ndim != 2:
            raise ValueError(f"mask must be a single-channel label image: {path}")
        return torch.from_numpy(labels)


def mask_metrics(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_mask: Path,
    target_mask: Path,
) -> dict[str, Any]:
    source_height, source_width = source_features.shape[:2]
    target_height, target_width = target_features.shape[:2]
    source_labels = _mask_grid(source_mask, source_height, source_width)
    target_labels = _mask_grid(target_mask, target_height, target_width)
    return mask_propagation_metrics(
        source_features,
        target_features,
        source_labels,
        target_labels,
    )


def _metric_value(record: dict[str, Any]) -> float:
    for key in ("mean_iou", "pck_1"):
        if record.get(key) is not None:
            return float(record[key])
    return -float(record.get("mse", float("inf")))


def _state_record(
    source: FrameFeatures,
    target: FrameFeatures,
    state: FastState,
    readout: Any,
    action_pair: tuple[str, str],
    sequence: SequenceInfo,
    target_slot: int,
    split: str,
    eta: float,
    ledger: list[dict[str, Any]],
    start: float,
) -> dict[str, Any]:
    target_z = readout.project(target.features)
    adapted_z = materialize(target_z, state)
    adapted = readout.restore(target.features, target_z, adapted_z)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "sequence": sequence.name,
        "split": split,
        "target_slot": target_slot,
        "eta": eta,
        "action_pair": list(action_pair),
        "pck_1": None,
        "pck_2": None,
        "mean_error": None,
        "n_points": None,
        "mean_iou": None,
        "foreground_iou": None,
        "n_objects": None,
        "mse": probe_mse(
            target_z,
            target.rgb,
            readout,
            state,
            torch.ones(target.features.shape[:2], dtype=torch.bool),
        ),
        "state_a_norm": float(state.a.norm()),
        "state_b_norm": float(state.b.norm()),
        "final_nonzero_matrix_count": int(state.a.norm() > 0) + int(state.b.norm() > 0),
        "committed_write_events": sum(
            {"OFF": 0, "A": 1, "B": 1, "ALL": 2}[entry["action"]]
            for entry in ledger
        ),
        "candidate_ledger": ledger,
        "runtime_seconds": time.perf_counter() - start,
        "diagnostic": False,
    }
    source_mask, target_mask = sequence.probe_masks()[0], sequence.probe_masks()[target_slot]
    if source_mask is not None and target_mask is not None:
        result.update(mask_metrics(source.features, adapted, source_mask, target_mask))
    if sequence.homographies and target_slot - 1 < len(sequence.homographies):
        homography_path = sequence.homographies[target_slot - 1]
        if homography_path is not None:
            from mcss.vision_probe.geometry import homography_correspondence, matching_metrics

            homography = torch.from_numpy(np.loadtxt(homography_path)).float()
            indices, ground_truth = homography_correspondence(
                homography,
                source.original_size,
                target.original_size,
                grid_size=int(source.features.shape[0]),
                image_size=source.image_size,
            )
            result.update(matching_metrics(source.features, adapted, indices, ground_truth))
    return result


def _run_pair(
    source: FrameFeatures,
    target: FrameFeatures,
    readout: Any,
    action_pair: tuple[str, str],
    eta: float,
) -> tuple[FastState, list[dict[str, Any]]]:
    del source
    state = FastState.zero(int(readout.decoder.shape[0]))
    target_z = readout.project(target.features)
    ledger: list[dict[str, Any]] = []
    for step, action in enumerate(action_pair):
        support, probe = masks(*target.features.shape[:2], step)
        candidates = proposals(target_z, target.rgb, readout, state, support, eta)
        ledger.append(
            {
                "step": step,
                "action": action,
                "support_count": int(support.sum()),
                "probe_count": int(probe.sum()),
                "candidate_evaluations": 0,
                "candidate_construction_count": 4,
                "candidate_probe_readouts": 0,
                "matrix_increment_computations": 2,
            }
        )
        state = candidates[action]
    return state, ledger


def _run_selector(
    source: FrameFeatures, target: FrameFeatures, readout: Any, eta: float
) -> tuple[FastState, tuple[str, str], list[dict[str, Any]]]:
    del source
    state = FastState.zero(int(readout.decoder.shape[0]))
    target_z = readout.project(target.features)
    chosen: list[str] = []
    ledger: list[dict[str, Any]] = []
    for step in range(2):
        support, probe = masks(*target.features.shape[:2], step)
        candidates = proposals(target_z, target.rgb, readout, state, support, eta)
        scores = {
            action: probe_mse(target_z, target.rgb, readout, candidate, probe)
            for action, candidate in candidates.items()
        }
        action = min(ACTIONS, key=lambda candidate: scores[candidate])
        chosen.append(action)
        ledger.append(
            {
                "step": step,
                "action": action,
                "support_count": int(support.sum()),
                "probe_count": int(probe.sum()),
                "candidate_evaluations": 4,
                "candidate_construction_count": 4,
                "candidate_probe_readouts": 4,
                "matrix_increment_computations": 2,
                "candidate_probe_mse": scores,
            }
        )
        state = candidates[action]
    return state, (chosen[0], chosen[1]), ledger


def _evaluate_sequence(
    sequence: SequenceInfo,
    features: dict[Path, FrameFeatures],
    readout: Any,
    split: str,
    eta: float,
    include_all: bool = True,
) -> list[dict[str, Any]]:
    frame_paths = sequence.probe_frames()
    source = features[frame_paths[0]]
    records: list[dict[str, Any]] = []
    for target_slot, frame_path in enumerate(frame_paths[1:], start=1):
        target = features[frame_path]
        trajectories: dict[tuple[str, str], dict[str, Any]] = {}
        for pair in itertools.product(ACTIONS, repeat=2):
            start = time.perf_counter()
            state, ledger = _run_pair(source, target, readout, pair, eta)
            trajectories[pair] = _state_record(
                source,
                target,
                state,
                readout,
                pair,
                sequence,
                target_slot,
                split,
                eta,
                ledger,
                start,
            )
            trajectories[pair]["method"] = "trajectory"
            trajectories[pair]["diagnostic"] = True
        records.extend(trajectories.values())
        fixed = {
            "OFFOFF": ("OFF", "OFF"),
            "AA": ("A", "A"),
            "BB": ("B", "B"),
            "ALLALL": ("ALL", "ALL"),
        }
        for method, pair in fixed.items():
            copy = dict(trajectories[pair])
            copy["method"] = method
            copy["diagnostic"] = False
            records.append(copy)
        start = time.perf_counter()
        state, pair, ledger = _run_selector(source, target, readout, eta)
        selected = _state_record(
            source, target, state, readout, pair, sequence, target_slot, split, eta, ledger, start
        )
        selected["method"] = "residual_selector"
        records.append(selected)
        constant = max((trajectories[(action, action)] for action in ACTIONS), key=_metric_value)
        constant = dict(constant)
        constant.update({"method": "oracle_bestconstantperpair", "diagnostic": True})
        records.append(constant)
        best = dict(max(trajectories.values(), key=_metric_value))
        best.update({"method": "oracle_bestsequenceperpair", "diagnostic": True})
        records.append(best)
    return records


def _select_global(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[float, tuple[str, str]], list[dict[str, Any]]] = {}
    for record in records:
        pair = tuple(record["action_pair"])
        groups.setdefault((float(record["eta"]), pair), []).append(record)
    candidates = [
        (sum(_metric_value(record) for record in values) / len(values), eta, pair)
        for (eta, pair), values in groups.items()
    ]
    score, eta, pair = max(candidates, key=lambda item: (item[0], -item[1], tuple(item[2])))
    return {
        "eta": eta,
        "action_pair": list(pair),
        "discovery_mean_metric": score,
        "selection_rule": "discovery_mean_metric_only",
    }


def _bootstrap(values: list[float], seed: int = 7, draws: int = 1000) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "lower95": None, "upper95": None}
    rng = random.Random(seed)
    samples = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(draws)
    ]
    samples.sort()
    return {
        "mean": float(sum(values) / len(values)),
        "lower95": samples[int(draws * 0.025)],
        "upper95": samples[int(draws * 0.975) - 1],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--backbone-repo",
        type=Path,
        default=PROJECT / "third_party" / "dinov2",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--fit-sequences", "--fit-sequences24", dest="fit_sequences", type=int, default=24
    )
    parser.add_argument(
        "--discovery-sequences",
        "--discovery-sequences12",
        dest="discovery_sequences",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--validation-sequences",
        "--validation-sequences16",
        dest="validation_sequences",
        type=int,
        default=16,
    )
    parser.add_argument("--eta", nargs="+", type=float, default=[0.5, 2.0, 5.0])
    return parser.parse_args()


def run(args: argparse.Namespace, extractor: Any | None = None) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_status(output, "running", "discover")
    try:
        sequences = discover_sequences(args.data_root)
        splits = split_sequences(
            sequences, args.fit_sequences, args.discovery_sequences, args.validation_sequences
        )
        assignments = {
            sequence.name: split for split, values in splits.items() for sequence in values
        }
        _write_json(
            output / "splits.json",
            {
                "schema": SCHEMA,
                "splits": {
                    split: [sequence.manifest_row(split) for sequence in values]
                    for split, values in splits.items()
                },
                "assignments": assignments,
            },
        )
        if extractor is None:
            from mcss.vision_probe.features import FrozenDinoExtractor

            extractor = FrozenDinoExtractor(
                repo_root=Path(args.backbone_repo),
                weights=Path(args.weights),
                device=args.device,
                size=args.size,
            )
        weight_hash = (
            _sha256_file(Path(args.weights))
            if Path(args.weights).is_file()
            else hashlib.sha256(str(args.weights).encode()).hexdigest()
        )
        source_hash = _module_hash()
        selected = tuple(
            sequence for split in ("fit", "discovery", "validation") for sequence in splits[split]
        )
        feature_map: dict[str, dict[Path, FrameFeatures]] = {}
        for index, sequence in enumerate(selected, start=1):
            feature_map[sequence.name] = {}
            for frame in sequence.probe_frames():
                feature_map[sequence.name][frame] = _cached_extract(
                    extractor, frame, output / "cache", weight_hash, source_hash, args.size
                )
            _write_status(
                output, "running", "extract", sequences_done=index, sequences_total=len(selected)
            )
        train_features = [
            feature_map[sequence.name][frame]
            for sequence in splits["fit"]
            for frame in sequence.probe_frames()
        ]
        if not train_features:
            raise ValueError("fit split is empty")
        feature_width = train_features[0].features.shape[-1]
        width = min(
            64,
            int(feature_width) - 1,
            sum(int(item.features.shape[0] * item.features.shape[1]) for item in train_features)
            - 1,
        )
        actual_width = max(1, width)
        readout = fit_readout(
            torch.cat([item.features for item in train_features]),
            torch.cat([item.rgb for item in train_features]),
            width=actual_width,
        )
        torch.save(
            {
                "mean": readout.mean,
                "basis": readout.basis,
                "scale": readout.scale,
                "decoder": readout.decoder,
                "bias": readout.bias,
                "width": actual_width,
                "schema": SCHEMA,
            },
            output / "readout.pt",
        )
        _write_status(output, "running", "discovery")
        raw_path = output / "raw.jsonl"
        raw_path.write_text("", encoding="utf-8")
        discovery_records: list[dict[str, Any]] = []
        with raw_path.open("a", encoding="utf-8") as handle:
            for eta in args.eta:
                for sequence in splits["discovery"]:
                    records = _evaluate_sequence(
                        sequence, feature_map[sequence.name], readout, "discovery", float(eta)
                    )
                    discovery_records.extend(records)
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        choice = _select_global(
            [record for record in discovery_records if record["method"] == "trajectory"]
        )
        _write_status(output, "running", "validation", selected=choice)
        validation_records: list[dict[str, Any]] = []
        with raw_path.open("a", encoding="utf-8") as handle:
            for sequence in splits["validation"]:
                records = _evaluate_sequence(
                    sequence,
                    feature_map[sequence.name],
                    readout,
                    "validation",
                    float(choice["eta"]),
                )
                pair = tuple(choice["action_pair"])
                selected_records = []
                for record in records:
                    if record["method"] != "trajectory" or tuple(record["action_pair"]) != pair:
                        continue
                    selected_record = dict(record)
                    selected_record["method"] = "bestgloballyfixed"
                    selected_record["diagnostic"] = False
                    selected_records.append(selected_record)
                records.extend(selected_records)
                validation_records.extend(records)
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        baseline = {
            (record["sequence"], record["target_slot"]): _metric_value(record)
            for record in validation_records
            if record["method"] == "OFFOFF"
        }
        per_sequence: dict[str, list[float]] = {}
        for record in validation_records:
            if record["method"] != "bestgloballyfixed":
                continue
            key = (record["sequence"], record["target_slot"])
            if key in baseline:
                per_sequence.setdefault(record["sequence"], []).append(
                    _metric_value(record) - baseline[key]
                )
        sequence_delta = {
            sequence: sum(values) / len(values)
            for sequence, values in per_sequence.items()
            if values
        }
        summary = {
            "schema": SCHEMA,
            "status": "complete",
            "exploratory": True,
            "official_benchmark": False,
            "split_counts": {split: len(values) for split, values in splits.items()},
            "feature_cache": {
                "weight_sha256": weight_hash,
                "model_source_sha256": source_hash,
                "size": args.size,
            },
            "discovery_selection": choice,
            "validation": {
                "bestgloballyfixed": {
                    "per_sequence_paired_delta": sequence_delta,
                    "bootstrap1000": _bootstrap(list(sequence_delta.values())),
                }
            },
            "fixed_methods": ["OFFOFF", "AA", "BB", "ALLALL", "residual_selector"],
            "oracle_methods": ["oracle_bestconstantperpair", "oracle_bestsequenceperpair"],
        }
        _write_json(output / "summary.json", summary)
        _write_status(output, "complete", "done", summary=str(output / "summary.json"))
        return summary
    except BaseException as error:
        _write_status(output, "failed", "error", error=repr(error))
        raise


def main() -> int:
    run(_parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
