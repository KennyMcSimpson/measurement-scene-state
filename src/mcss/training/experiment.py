"""Resumable, train-only optimization for the offline dynamic carrier."""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
from pathlib import Path

import torch
from torch import Tensor

from mcss.data.episodes import OnlineEpisodeSource
from mcss.data.pilot_provenance import (
    DEFAULT_PREPARED_ROOT,
    PilotEntryProvenance,
    validate_pilot_entries,
    write_pilot_provenance,
)
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import load_dynamic_checkpoint, save_dynamic_checkpoint
from mcss.dynamic.config import CarrierConfig, WriteConfig
from mcss.dynamic.types import Action, OnlineObservation
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.measurements import FixedMeasurementRenderer
from mcss.training.grounded import grounded_measurement_loss, paired_context_measurement_loss
from mcss.training.supervision import TrainingSupervision
from mcss.training.write_unroll import unroll_observed_episode

EPISODE_INDEX_SCHEMA = "mcss.dynamic.episode_index.v1"
EXPERIMENT_SCHEMA = "mcss.dynamic.training_experiment.v1"
RESUME_SCHEMA = "mcss.dynamic.training_resume.v1"
_INDEX_FIELDS = frozenset({"schema_version", "episodes"})
_ACTION_VALUES = (Action.FUSE.value, Action.COMPLETE.value, Action.ALL.value)
_ALL_ACTION_VALUES = tuple(action.value for action in Action)
_PROJECT = Path(__file__).resolve().parents[3]
_IMPLEMENTATION_PATHS = (
    _PROJECT / "src/mcss/training/experiment.py",
    _PROJECT / "src/mcss/training/grounded.py",
    _PROJECT / "src/mcss/training/supervision.py",
    _PROJECT / "src/mcss/training/write_unroll.py",
    _PROJECT / "src/mcss/dynamic/carrier.py",
    _PROJECT / "src/mcss/dynamic/cache.py",
    _PROJECT / "src/mcss/dynamic/checkpoint.py",
    _PROJECT / "src/mcss/dynamic/config.py",
    _PROJECT / "src/mcss/dynamic/feedback.py",
    _PROJECT / "src/mcss/dynamic/types.py",
    _PROJECT / "src/mcss/dynamic/write_rule.py",
    _PROJECT / "scripts/train_dynamic_experiment.py",
    _PROJECT / "scripts/compile_dynamic_training_episodes.py",
)


@dataclass(frozen=True)
class DynamicExperimentConfig:
    """Protocol and optimization settings for the train-only experiment.

    The default values retain the original 4+8+4, fixed-action experiment.  The
    segment lengths and B trajectory schedule are explicit so a complete run can
    use longer windows without changing the trainer implementation.
    """

    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    write: WriteConfig = field(default_factory=WriteConfig)
    warmup_steps: int = 4
    stream_steps: int = 8
    query_count: int = 4
    stage_a_steps: int = 1000
    stage_b_steps: int = 600
    stage_a_learning_rate: float = 1e-3
    stage_b_learning_rate: float = 3e-4
    renderer_samples: int = 16
    ray_chunk_size: int = 2048
    gradient_clip_norm: float = 1.0
    checkpoint_interval: int = 100
    status_interval: int = 10
    cache_capacity: int = 4
    seed: int = 20260919
    torch_num_threads: int = 4
    action_schedule_mode: str = "legacy_fixed"
    stage_b_actions: tuple[str, ...] = _ACTION_VALUES
    stage_b_trajectories: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in ("warmup_steps", "stream_steps", "query_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stage_a_steps < 0 or self.stage_b_steps < 0:
            raise ValueError("stage step counts must be nonnegative")
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                self.stage_a_learning_rate,
                self.stage_b_learning_rate,
                self.gradient_clip_norm,
            )
        ):
            raise ValueError("learning rates and gradient_clip_norm must be finite and positive")
        if self.renderer_samples < 2 or self.ray_chunk_size < 1:
            raise ValueError("renderer_samples must be at least two and ray_chunk_size positive")
        if self.checkpoint_interval < 1 or self.status_interval < 1:
            raise ValueError("checkpoint_interval and status_interval must be positive")
        if self.cache_capacity < 1 or self.torch_num_threads < 1:
            raise ValueError("cache_capacity and torch_num_threads must be positive")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        mode = str(self.action_schedule_mode).strip().lower()
        mode_aliases = {
            "legacy": "legacy_fixed",
            "fixed": "legacy_fixed",
            "legacy_fixed": "legacy_fixed",
            "mixed": "mixed",
            "mixed_cycle": "mixed",
            "mixed_cycles": "mixed",
        }
        if mode not in mode_aliases:
            raise ValueError("action_schedule_mode must be legacy_fixed or mixed")
        object.__setattr__(self, "action_schedule_mode", mode_aliases[mode])
        actions = _normalize_actions(self.stage_b_actions, "stage_b_actions")
        if not actions:
            raise ValueError("stage_b_actions must contain at least one action")
        object.__setattr__(self, "stage_b_actions", actions)
        trajectories = _normalize_trajectories(
            self.stage_b_trajectories,
            stream_steps=self.stream_steps,
        )
        object.__setattr__(self, "stage_b_trajectories", trajectories)

    @classmethod
    def from_json(cls, path: str | Path) -> DynamicExperimentConfig:
        return load_dynamic_experiment_config(path)


def _normalize_action(value: object, name: str) -> str:
    if isinstance(value, Action):
        action = value.value
    elif isinstance(value, str):
        action = value.strip().upper()
    else:
        raise ValueError(f"{name} contains a non-string action")
    if action not in _ALL_ACTION_VALUES:
        raise ValueError(f"{name} contains unsupported action {action!r}")
    return action


def _normalize_actions(
    value: object, name: str, *, unique: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of action names")
    normalized: list[str] = []
    for item in value:
        action = _normalize_action(item, name)
        if not unique or action not in normalized:
            normalized.append(action)
    return tuple(normalized)


def _normalize_trajectories(
    value: object, *, stream_steps: int
) -> tuple[tuple[str, ...], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("stage_b_trajectories must be a list of action lists")
    trajectories: list[tuple[str, ...]] = []
    for index, trajectory in enumerate(value):
        actions = _normalize_actions(
            trajectory, f"stage_b_trajectories[{index}]", unique=False
        )
        if len(actions) != stream_steps:
            raise ValueError(
                f"stage_b_trajectories[{index}] must contain exactly {stream_steps} actions"
            )
        if actions not in trajectories:
            trajectories.append(actions)
    return tuple(trajectories)


def _normalize_nested_sequences(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize_nested_sequences(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_normalize_nested_sequences(item) for item in value)
    return value


def dynamic_experiment_config_from_mapping(
    value: Mapping[str, object], *, overrides: Mapping[str, object] | None = None
) -> DynamicExperimentConfig:
    """Build a validated config from a JSON-compatible nested mapping."""

    if not isinstance(value, Mapping):
        raise ValueError("dynamic experiment config must be an object")
    payload = dict(value.get("config", value))
    if not isinstance(payload, dict):
        raise ValueError("dynamic experiment config.config must be an object")
    if overrides:
        payload.update(dict(overrides))
    normalized = _normalize_nested_sequences(payload)
    if not isinstance(normalized, dict):
        raise ValueError("dynamic experiment config must be an object")
    carrier_value = normalized.get("carrier", {})
    write_value = normalized.get("write", {})
    if not isinstance(carrier_value, Mapping) or not isinstance(write_value, Mapping):
        raise ValueError("config carrier and write values must be objects")
    carrier = CarrierConfig(**dict(carrier_value))
    write = WriteConfig(**dict(write_value))
    normalized["carrier"] = carrier
    normalized["write"] = write
    # Accept the concise names used by command-line JSON files as aliases.
    aliases = {
        "warmup_length": "warmup_steps",
        "stream_length": "stream_steps",
        "query_length": "query_count",
        "b_action_values": "stage_b_actions",
        "action_values": "stage_b_actions",
        "fixed_actions": "stage_b_actions",
        "action_trajectories": "stage_b_trajectories",
        "mixed_trajectories": "stage_b_trajectories",
        "trajectory_patterns": "stage_b_trajectories",
        "schedule_mode": "action_schedule_mode",
    }
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    accepted = {field.name for field in fields(DynamicExperimentConfig)}
    unknown = sorted(set(normalized) - accepted - set(aliases))
    if unknown:
        raise ValueError(f"unknown dynamic experiment config fields: {unknown}")
    return DynamicExperimentConfig(
        **{key: item for key, item in normalized.items() if key in accepted}
    )


def load_dynamic_experiment_config(path: str | Path) -> DynamicExperimentConfig:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid dynamic experiment config JSON: {source}") from error
    return dynamic_experiment_config_from_mapping(value)


@dataclass(frozen=True)
class CachedTrainEpisode:
    """One fully observed CPU episode and its training-private query labels."""

    episode_id: str
    scene_id: str
    warmup: tuple[OnlineObservation, ...]
    stream: tuple[OnlineObservation, ...]
    supervision: TrainingSupervision
    valid_depth_fractions: tuple[float, ...]

    @property
    def observations(self) -> tuple[OnlineObservation, ...]:
        return (*self.warmup, *self.stream)


class CpuEpisodeCache:
    """Small LRU cache that decodes complete train episodes only on CPU."""

    def __init__(
        self,
        entries: Sequence[Mapping[str, object]],
        capacity: int,
        *,
        warmup_steps: int = 4,
        stream_steps: int = 8,
        query_count: int = 4,
    ) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        for name, value in (
            ("warmup_steps", warmup_steps),
            ("stream_steps", stream_steps),
            ("query_count", query_count),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._entries = {str(entry["episode_id"]): dict(entry) for entry in entries}
        if len(self._entries) != len(entries):
            raise ValueError("experiment index contains duplicate episode_id values")
        self._capacity = capacity
        self._warmup_steps = warmup_steps
        self._stream_steps = stream_steps
        self._query_count = query_count
        self._cache: OrderedDict[str, CachedTrainEpisode] = OrderedDict()

    def get(self, episode_id: str) -> CachedTrainEpisode:
        if episode_id in self._cache:
            self._cache.move_to_end(episode_id)
            return self._cache[episode_id]
        try:
            entry = self._entries[episode_id]
        except KeyError as error:
            raise KeyError(f"unknown train episode {episode_id!r}") from error
        cached = self._load(entry)
        self._cache[episode_id] = cached
        self._cache.move_to_end(episode_id)
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return cached

    def _load(self, entry: Mapping[str, object]) -> CachedTrainEpisode:
        source = OnlineEpisodeSource.from_file(
            str(entry["online_manifest"]),
            device="cpu",
            episode_id=str(entry["episode_id"]),
        )
        if source.split_id != "train":
            raise ValueError("dynamic experiment may only decode training sources")
        if source.scene_id != entry["scene_id"] or source.query_vault_id != entry["query_vault_id"]:
            raise ValueError("online source identity does not match the train index")
        if (
            entry["stream_steps"] != self._stream_steps
            or entry["query_count"] != self._query_count
        ):
            raise ValueError(
                "dynamic experiment index segment lengths do not match the configured "
                f"stream/query counts ({self._stream_steps}/{self._query_count})"
            )
        if source.remaining_steps != self._stream_steps:
            raise ValueError(
                "dynamic experiment requires exactly "
                f"{self._stream_steps} unread stream frames"
            )

        warmup = source.warmup()
        stream = tuple(source.next_observation() for _ in range(self._stream_steps))
        observations = (*warmup, *stream)
        observed_ids = tuple(observation.frame_id for observation in observations)
        if len(warmup) != self._warmup_steps or len(observations) != (
            self._warmup_steps + self._stream_steps
        ):
            raise ValueError(
                "dynamic experiment warmup/stream lengths do not match the configured "
                f"{self._warmup_steps}/{self._stream_steps}"
            )
        if observed_ids != tuple(sorted(observed_ids)) or len(set(observed_ids)) != len(
            observed_ids
        ):
            raise ValueError("dynamic experiment requires strictly ordered observations")
        if source.remaining_steps != 0:
            raise ValueError("dynamic experiment did not consume the complete online stream")

        supervision = TrainingSupervision.from_query_file(
            str(entry["query_manifest"]),
            episode_id=source.episode_id,
            scene_id=source.scene_id,
            query_vault_id=source.query_vault_id,
            context_frame_ids=observed_ids,
            device="cpu",
        )
        query_ids = supervision.frame_ids
        if len(query_ids) != self._query_count or query_ids != tuple(sorted(query_ids)):
            raise ValueError(
                "dynamic experiment requires "
                f"{self._query_count} strictly ordered query views"
            )
        if set(observed_ids) & set(query_ids):
            raise ValueError("dynamic experiment queries must be disjoint from observations")
        if query_ids[0] <= observed_ids[-1]:
            raise ValueError("dynamic experiment queries must be later than observed frames")
        valid = torch.isfinite(supervision.query_depth) & (supervision.query_depth > 0)
        if not valid.any():
            raise ValueError(f"training episode {source.episode_id} has no valid depth pixels")
        sanitized_depth = torch.where(
            valid, supervision.query_depth, torch.zeros_like(supervision.query_depth)
        )
        fractions = tuple(
            float(valid[:, view : view + 1].float().mean()) for view in range(self._query_count)
        )
        return CachedTrainEpisode(
            episode_id=source.episode_id,
            scene_id=source.scene_id,
            warmup=warmup,
            stream=stream,
            supervision=replace(supervision, query_depth=sanitized_depth),
            valid_depth_fractions=fractions,
        )


class _PermutationCycle:
    """An independently seeded, resumable permutation cycle."""

    def __init__(self, values: Sequence[str | int], generator: torch.Generator) -> None:
        self.values = tuple(values)
        if not self.values or len(set(self.values)) != len(self.values):
            raise ValueError("permutation cycle values must be nonempty and unique")
        self.generator = generator
        self.order: list[str | int] = []
        self.cursor = 0
        self.cycles = 0

    def next(self) -> str | int:
        if self.cursor == len(self.order):
            indices = torch.randperm(len(self.values), generator=self.generator).tolist()
            self.order = [self.values[index] for index in indices]
            self.cursor = 0
            self.cycles += 1
        value = self.order[self.cursor]
        self.cursor += 1
        return value

    def state_dict(self) -> dict[str, object]:
        return {
            "values": list(self.values),
            "generator_state": self.generator.get_state(),
            "order": list(self.order),
            "cursor": self.cursor,
            "cycles": self.cycles,
        }

    @classmethod
    def from_state(
        cls,
        values: Sequence[str | int],
        generator: torch.Generator,
        state: Mapping[str, object],
    ) -> _PermutationCycle:
        if list(values) != state.get("values"):
            raise ValueError("resume permutation values do not match the experiment index")
        order = state.get("order")
        cursor = state.get("cursor")
        cycles = state.get("cycles")
        generator_state = state.get("generator_state")
        if (
            not isinstance(order, list)
            or any(value not in values for value in order)
            or len(order) not in (0, len(values))
            or len(set(order)) != len(order)
            or type(cursor) is not int
            or cursor < 0
            or cursor > len(order)
            or type(cycles) is not int
            or cycles < 0
            or not isinstance(generator_state, Tensor)
        ):
            raise ValueError("resume permutation state is invalid")
        generator.set_state(generator_state.cpu())
        result = cls(values, generator)
        result.order = list(order)
        result.cursor = cursor
        result.cycles = cycles
        return result


def run_dynamic_experiment(
    episode_index: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device,
    config: DynamicExperimentConfig | None = None,
    resume: bool = False,
    prepared_root: str | Path = DEFAULT_PREPARED_ROOT,
    init_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    """Run the configured A-then-B train-only protocol, or resume it exactly."""

    config = config or DynamicExperimentConfig()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA experiment requested but CUDA is unavailable")
    torch.set_num_threads(config.torch_num_threads)

    index_path = Path(episode_index).resolve()
    entries = load_train_episode_index(index_path)
    index_sha256 = _sha256_file(index_path)
    # This validates every manifest and reference before CpuEpisodeCache decodes RGB/depth.
    provenances = validate_pilot_entries(entries, prepared_root=prepared_root)
    source_sha256 = _canonical_sha256([item.to_dict() for item in provenances])
    config_payload = _config_payload(config)
    config_sha256 = _canonical_sha256(config_payload)
    implementation_sources = _implementation_source_hashes()
    implementation_sha256 = _canonical_sha256(implementation_sources)
    output = Path(output_dir).resolve()
    resume_path = output / "resume.pt"
    initial_checkpoint_sha256 = _initial_checkpoint_hash(
        init_checkpoint,
        output=output,
        resume=resume,
    )

    _prepare_output(output, resume=resume)
    if resume:
        resume_payload = _load_resume(resume_path, target_device)
        if init_checkpoint is None:
            saved_initial_hash = resume_payload.get("initial_checkpoint_sha256")
            if saved_initial_hash is not None and not isinstance(saved_initial_hash, str):
                raise ValueError("resume initial_checkpoint_sha256 must be a string or null")
            initial_checkpoint_sha256 = saved_initial_hash
        _validate_resume(
            resume_payload,
            config_sha256=config_sha256,
            index_sha256=index_sha256,
            source_sha256=source_sha256,
            implementation_sha256=implementation_sha256,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            config=config,
            device=target_device,
        )
    else:
        _atomic_json(output / "config.json", config_payload)
        _write_manifest_provenance(output / "manifest_provenance.json", provenances)
        _atomic_json(
            output / "run_metadata.json",
            {
                "schema_version": EXPERIMENT_SCHEMA,
                "episode_index": str(index_path),
                "episode_index_sha256": index_sha256,
                "validated_manifest_source_sha256": source_sha256,
                "implementation_source_sha256": implementation_sha256,
                "implementation_sources": implementation_sources,
                "manifest_provenance": str(output / "manifest_provenance.json"),
                "initial_checkpoint": (
                    None if init_checkpoint is None else str(Path(init_checkpoint).resolve())
                ),
                "initial_checkpoint_sha256": initial_checkpoint_sha256,
            },
        )
        resume_payload = None

    cache = CpuEpisodeCache(
        entries,
        config.cache_capacity,
        warmup_steps=config.warmup_steps,
        stream_steps=config.stream_steps,
        query_count=config.query_count,
    )
    scene_ids = tuple(str(entry["episode_id"]) for entry in entries)
    scene_cycle, query_cycle, action_cycle = _make_cycles(
        scene_ids, config.seed, resume_payload, config
    )
    renderer = FixedMeasurementRenderer(
        n_samples=config.renderer_samples,
        ray_chunk_size=config.ray_chunk_size,
    ).to(target_device)
    started_at = time.perf_counter()

    if resume_payload is None:
        torch.manual_seed(config.seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        if init_checkpoint is None:
            carrier = DynamicSceneCarrier(config.carrier).to(target_device)
            write_rule = DirectWriteRule(config.carrier, config.write).to(target_device)
        else:
            carrier, write_rule, checkpoint_metadata = load_dynamic_checkpoint(
                init_checkpoint, target_device
            )
            if carrier.config != config.carrier or write_rule.write_config != config.write:
                raise ValueError(
                    "initial checkpoint CarrierConfig/WriteConfig does not match the experiment"
                )
        _set_trainable(carrier, write_rule, phase="A")
        optimizer = torch.optim.Adam(carrier.parameters(), lr=config.stage_a_learning_rate)
        phase = "A"
        phase_step = 0
        global_step = 0
        stage_steps = {"A": 0, "B": 0}
        last_history_record: dict[str, object] | None = None
        _save_checkpoint(
            output / "initial.pt",
            carrier,
            write_rule,
            phase="initial",
            config_sha256=config_sha256,
            index_sha256=index_sha256,
            source_sha256=source_sha256,
            implementation_sha256=implementation_sha256,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
        )
        _save_resume(
            resume_path,
            carrier=carrier,
            write_rule=write_rule,
            optimizer=optimizer,
            optimizer_phase="A",
            phase=phase,
            phase_step=phase_step,
            global_step=global_step,
            stage_steps=stage_steps,
            config_sha256=config_sha256,
            index_sha256=index_sha256,
            source_sha256=source_sha256,
            implementation_sha256=implementation_sha256,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            device=target_device,
            scene_cycle=scene_cycle,
            query_cycle=query_cycle,
            action_cycle=action_cycle,
            last_history_record=last_history_record,
        )
        _write_status(output / "status.json", "running", phase, phase_step, global_step)
    else:
        phase = str(resume_payload["phase"])
        phase_step = int(resume_payload["phase_step"])
        global_step = int(resume_payload["global_step"])
        stage_steps = dict(resume_payload["stage_steps"])
        last_history_record = resume_payload.get("last_history_record")
        carrier, write_rule = _restore_modules(config, resume_payload, target_device)
        optimizer_phase = str(resume_payload["optimizer_phase"])
        if phase == "A":
            _set_trainable(carrier, write_rule, phase="A")
            optimizer = torch.optim.Adam(carrier.parameters(), lr=config.stage_a_learning_rate)
        elif phase in {"B", "complete"}:
            _set_trainable(carrier, write_rule, phase="B")
            optimizer = torch.optim.Adam(
                [*carrier.parameters(), *write_rule.parameters()], lr=config.stage_b_learning_rate
            )
        else:
            raise ValueError("resume phase must be A, B, or complete")
        if optimizer_phase not in {"A", "B"}:
            raise ValueError("resume optimizer_phase must be A or B")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        _restore_global_rng(resume_payload, target_device)
        _reconcile_history(output / "history.jsonl", global_step, last_history_record)

    try:
        if phase == "complete":
            return _completed_report(output)

        while phase == "A" and phase_step < config.stage_a_steps:
            episode_id = str(scene_cycle.next())
            query_index = int(query_cycle.next())
            episode = cache.get(episode_id)
            query = _select_query(episode, query_index, target_device)
            contexts = _to_device_observations(episode.observations, target_device)
            even_context = tuple(contexts[index] for index in range(0, len(contexts), 2))
            odd_context = tuple(contexts[index] for index in range(1, len(contexts), 2))
            if not even_context or not odd_context:
                raise ValueError("phase A requires at least two observed frames")
            optimizer.zero_grad(set_to_none=True)
            paired = paired_context_measurement_loss(
                carrier,
                even_context,
                odd_context,
                (query["frame_id"],),
                query["cameras"],
                query["rgb"],
                query["depth"],
                renderer,
            )
            gradients = _backward_step(
                paired.loss,
                carrier,
                write_rule,
                optimizer,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            phase_step += 1
            global_step += 1
            stage_steps["A"] = phase_step
            record = _history_record(
                global_step=global_step,
                phase="A",
                phase_step=phase_step,
                scene_id=episode.scene_id,
                query_frame_id=query["frame_id"],
                action=Action.OFF.value,
                loss=paired.loss,
                terms=paired.terms,
                gradients=gradients,
                seconds=time.perf_counter() - started_at,
                valid_depth_fraction=query["valid_depth_fraction"],
            )
            _persist_step(
                output,
                carrier=carrier,
                write_rule=write_rule,
                optimizer=optimizer,
                optimizer_phase="A",
                phase="A",
                phase_step=phase_step,
                global_step=global_step,
                stage_steps=stage_steps,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                source_sha256=source_sha256,
                implementation_sha256=implementation_sha256,
                initial_checkpoint_sha256=initial_checkpoint_sha256,
                device=target_device,
                scene_cycle=scene_cycle,
                query_cycle=query_cycle,
                action_cycle=action_cycle,
                record=record,
                checkpoint_interval=config.checkpoint_interval,
                status_interval=config.status_interval,
            )

        if phase == "A":
            _save_checkpoint(
                output / "phase_a_final.pt",
                carrier,
                write_rule,
                phase="phase_a_final",
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                source_sha256=source_sha256,
                implementation_sha256=implementation_sha256,
                initial_checkpoint_sha256=initial_checkpoint_sha256,
            )
            carrier, write_rule, _ = load_dynamic_checkpoint(
                output / "phase_a_final.pt", target_device
            )
            _set_trainable(carrier, write_rule, phase="B")
            optimizer = torch.optim.Adam(
                [*carrier.parameters(), *write_rule.parameters()], lr=config.stage_b_learning_rate
            )
            phase = "B"
            phase_step = 0
            _save_resume(
                resume_path,
                carrier=carrier,
                write_rule=write_rule,
                optimizer=optimizer,
                optimizer_phase="B",
                phase=phase,
                phase_step=phase_step,
                global_step=global_step,
                stage_steps=stage_steps,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                source_sha256=source_sha256,
                implementation_sha256=implementation_sha256,
                initial_checkpoint_sha256=initial_checkpoint_sha256,
                device=target_device,
                scene_cycle=scene_cycle,
                query_cycle=query_cycle,
                action_cycle=action_cycle,
                last_history_record=None,
            )
            _write_status(output / "status.json", "running", phase, phase_step, global_step)

        while phase == "B" and phase_step < config.stage_b_steps:
            episode_id = str(scene_cycle.next())
            query_index = int(query_cycle.next())
            trajectory = _trajectory_from_label(str(action_cycle.next()), config)
            episode = cache.get(episode_id)
            query = _select_query(episode, query_index, target_device)
            warmup = _to_device_observations(episode.warmup, target_device)
            stream = _to_device_observations(episode.stream, target_device)
            optimizer.zero_grad(set_to_none=True)
            state, _, _, anchor_c2w = unroll_observed_episode(
                carrier,
                write_rule,
                warmup,
                stream,
                tuple(Action(action_name) for action_name in trajectory),
                f"{episode.episode_id}:B:{global_step + 1}",
            )
            loss, terms = grounded_measurement_loss(
                state,
                query["cameras"],
                query["rgb"],
                query["depth"],
                anchor_c2w,
                renderer,
            )
            gradients = _backward_step(
                loss,
                carrier,
                write_rule,
                optimizer,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            phase_step += 1
            global_step += 1
            stage_steps["B"] = phase_step
            record = _history_record(
                global_step=global_step,
                phase="B",
                phase_step=phase_step,
                scene_id=episode.scene_id,
                query_frame_id=query["frame_id"],
                action=_legacy_action_field(trajectory),
                actions=trajectory if config.action_schedule_mode == "mixed" else None,
                loss=loss,
                terms=terms,
                gradients=gradients,
                seconds=time.perf_counter() - started_at,
                valid_depth_fraction=query["valid_depth_fraction"],
            )
            _persist_step(
                output,
                carrier=carrier,
                write_rule=write_rule,
                optimizer=optimizer,
                optimizer_phase="B",
                phase="B",
                phase_step=phase_step,
                global_step=global_step,
                stage_steps=stage_steps,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                source_sha256=source_sha256,
                implementation_sha256=implementation_sha256,
                initial_checkpoint_sha256=initial_checkpoint_sha256,
                device=target_device,
                scene_cycle=scene_cycle,
                query_cycle=query_cycle,
                action_cycle=action_cycle,
                record=record,
                checkpoint_interval=config.checkpoint_interval,
                status_interval=config.status_interval,
            )

        if phase == "B":
            _save_checkpoint(
                output / "phase_b_final.pt",
                carrier,
                write_rule,
                phase="phase_b_final",
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                source_sha256=source_sha256,
                implementation_sha256=implementation_sha256,
                initial_checkpoint_sha256=initial_checkpoint_sha256,
            )
            phase = "complete"
            _save_resume(
                resume_path,
                carrier=carrier,
                write_rule=write_rule,
                optimizer=optimizer,
                optimizer_phase="B",
                phase=phase,
                phase_step=phase_step,
                global_step=global_step,
                stage_steps=stage_steps,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                source_sha256=source_sha256,
                implementation_sha256=implementation_sha256,
                initial_checkpoint_sha256=initial_checkpoint_sha256,
                device=target_device,
                scene_cycle=scene_cycle,
                query_cycle=query_cycle,
                action_cycle=action_cycle,
                last_history_record=None,
            )
            _write_status(output / "status.json", "complete", phase, phase_step, global_step)
        return _completed_report(output)
    except Exception as error:
        _atomic_json(
            output / "status.json",
            {
                "schema_version": EXPERIMENT_SCHEMA,
                "status": "failed",
                "phase": phase,
                "phase_step": phase_step,
                "global_step": global_step,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def load_train_episode_index(path: str | Path) -> tuple[dict[str, object], ...]:
    """Load the exact all-train index schema before any dataset decoding occurs."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid dynamic train index JSON: {source}") from error
    if not isinstance(value, dict) or frozenset(value) != _INDEX_FIELDS:
        raise ValueError("dynamic train index root fields must be schema_version and episodes")
    if value["schema_version"] != EPISODE_INDEX_SCHEMA:
        raise ValueError(f"expected dynamic train index schema {EPISODE_INDEX_SCHEMA}")
    episodes = value["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dynamic train index requires a nonempty episode list")
    if not all(isinstance(entry, dict) for entry in episodes):
        raise ValueError("dynamic train index episodes must be objects")
    if any(entry.get("split_id") != "train" for entry in episodes):
        raise ValueError("dynamic experiment rejects every non-train index entry")
    return tuple(dict(entry) for entry in episodes)


def _make_cycles(
    scene_ids: tuple[str, ...],
    seed: int,
    resume: Mapping[str, object] | None,
    config: DynamicExperimentConfig,
) -> tuple[_PermutationCycle, _PermutationCycle, _PermutationCycle]:
    generators = [torch.Generator(device="cpu") for _ in range(3)]
    if resume is None:
        for offset, generator in enumerate(generators, start=1):
            generator.manual_seed(seed + offset)
        return (
            _PermutationCycle(scene_ids, generators[0]),
            _PermutationCycle(tuple(range(config.query_count)), generators[1]),
            _PermutationCycle(_schedule_values(config), generators[2]),
        )
    schedules = resume.get("schedules")
    if not isinstance(schedules, Mapping):
        raise ValueError("resume schedules must be a mapping")
    try:
        return (
            _PermutationCycle.from_state(scene_ids, generators[0], _mapping(schedules["scene"])),
            _PermutationCycle.from_state(
                tuple(range(config.query_count)), generators[1], _mapping(schedules["query"])
            ),
            _PermutationCycle.from_state(
                _schedule_values(config), generators[2], _mapping(schedules["action"])
            ),
        )
    except KeyError as error:
        raise ValueError("resume schedules are missing a required cycle") from error


def _select_query(
    episode: CachedTrainEpisode, query_index: int, device: torch.device
) -> dict[str, object]:
    if query_index not in range(len(episode.valid_depth_fractions)):
        raise ValueError("query_index must select one of the configured query views")
    fraction = episode.valid_depth_fractions[query_index]
    if fraction <= 0.0:
        raise ValueError(
            "selected query frame "
            f"{episode.supervision.frame_ids[query_index]} has no valid depth pixels"
        )
    index = torch.tensor([query_index], dtype=torch.long)
    supervision = episode.supervision
    return {
        "frame_id": supervision.frame_ids[query_index],
        "cameras": supervision.query_cameras.select_views(index).to(device),
        "rgb": supervision.query_rgb[:, query_index : query_index + 1].to(device),
        "depth": supervision.query_depth[:, query_index : query_index + 1].to(device),
        "valid_depth_fraction": fraction,
    }


def _schedule_values(config: DynamicExperimentConfig) -> tuple[str, ...]:
    """Return balanced, independently shuffled labels for the B trajectory cycle."""

    if config.action_schedule_mode == "legacy_fixed":
        return tuple(config.stage_b_actions)
    fixed = tuple(f"fixed:{action}" for action in config.stage_b_actions)
    trajectories = tuple(
        f"trajectory:{index}" for index in range(len(config.stage_b_trajectories))
    )
    if fixed and trajectories:
        import math as _math

        cycle_size = _math.lcm(len(fixed), len(trajectories))
        fixed = tuple(
            f"{fixed[index % len(fixed)]}#{index // len(fixed)}"
            for index in range(cycle_size)
        )
        trajectories = tuple(
            f"{trajectories[index % len(trajectories)]}#{index // len(trajectories)}"
            for index in range(cycle_size)
        )
    values = (*fixed, *trajectories)
    if not values:
        raise ValueError("mixed action schedule requires fixed actions or trajectories")
    return values


def _trajectory_from_label(label: str, config: DynamicExperimentConfig) -> tuple[str, ...]:
    if config.action_schedule_mode == "legacy_fixed":
        action = Action(label)
        return (action.value,) * config.stream_steps
    base_label = label.split("#", 1)[0]
    if base_label.startswith("fixed:"):
        action = Action(base_label.removeprefix("fixed:"))
        return (action.value,) * config.stream_steps
    if base_label.startswith("trajectory:"):
        try:
            index = int(base_label.removeprefix("trajectory:"))
        except ValueError as error:
            raise ValueError(f"invalid action trajectory label: {label!r}") from error
        try:
            return config.stage_b_trajectories[index]
        except IndexError as error:
            raise ValueError(f"unknown action trajectory label: {label!r}") from error
    # Permit a legacy resume payload to be interpreted under an explicitly mixed config.
    if label in _ALL_ACTION_VALUES:
        return (Action(label).value,) * config.stream_steps
    raise ValueError(f"invalid action schedule label: {label!r}")


def _legacy_action_field(trajectory: Sequence[str]) -> str:
    values = tuple(trajectory)
    if not values:
        raise ValueError("action trajectory must be nonempty")
    return values[0] if len(set(values)) == 1 else "MIXED"


def _to_device_observations(
    observations: Sequence[OnlineObservation], device: torch.device
) -> tuple[OnlineObservation, ...]:
    return tuple(
        OnlineObservation(
            scene_id=observation.scene_id,
            frame_id=observation.frame_id,
            rgb=observation.rgb.to(device),
            camera=observation.camera.to(device),
        )
        for observation in observations
    )


def _set_trainable(carrier, write_rule, *, phase: str) -> None:
    if phase not in {"A", "B"}:
        raise ValueError("trainable phase must be A or B")
    for parameter in carrier.parameters():
        parameter.requires_grad_(True)
    for parameter in write_rule.parameters():
        parameter.requires_grad_(phase == "B")


def _backward_step(
    loss: Tensor,
    carrier,
    write_rule,
    optimizer,
    *,
    gradient_clip_norm: float,
) -> dict[str, float]:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("dynamic experiment produced a nonfinite scalar loss")
    loss.backward()
    carrier_norm = _gradient_norm(carrier.parameters(), "carrier")
    write_norm = _gradient_norm(write_rule.parameters(), "write rule")
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return {"carrier_gradient_norm": carrier_norm, "write_gradient_norm": write_norm}


def _gradient_norm(parameters, name: str) -> float:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f"{name} gradients are nonfinite")
    squared_norm = sum(gradient.detach().float().square().sum() for gradient in gradients)
    return float(torch.sqrt(squared_norm).cpu())


def _history_record(
    *,
    global_step: int,
    phase: str,
    phase_step: int,
    scene_id: str,
    query_frame_id: object,
    action: str,
    loss: Tensor,
    terms: Mapping[str, Tensor],
    gradients: Mapping[str, float],
    seconds: float,
    valid_depth_fraction: object,
    actions: Sequence[str] | None = None,
) -> dict[str, object]:
    if not isinstance(query_frame_id, int) or not isinstance(valid_depth_fraction, float):
        raise TypeError("history query metadata must be normalized")
    numeric_terms = {name: float(value.detach().cpu()) for name, value in sorted(terms.items())}
    if not all(math.isfinite(value) for value in numeric_terms.values()):
        raise RuntimeError("dynamic experiment produced nonfinite loss terms")
    record = {
        "global_step": global_step,
        "phase": phase,
        "phase_step": phase_step,
        "scene_id": scene_id,
        "query_frame_id": query_frame_id,
        "action": action,
        "loss": float(loss.detach().cpu()),
        "terms": numeric_terms,
        "carrier_gradient_norm": gradients["carrier_gradient_norm"],
        "write_gradient_norm": gradients["write_gradient_norm"],
        "seconds": float(seconds),
        "valid_depth_fraction": valid_depth_fraction,
    }
    if actions is not None:
        normalized_actions = tuple(_normalize_action(item, "history actions") for item in actions)
        if not normalized_actions:
            raise ValueError("history actions must be nonempty")
        record["actions"] = list(normalized_actions)
    return record


def _persist_step(
    output: Path,
    *,
    carrier,
    write_rule,
    optimizer,
    optimizer_phase: str,
    phase: str,
    phase_step: int,
    global_step: int,
    stage_steps: Mapping[str, int],
    config_sha256: str,
    index_sha256: str,
    source_sha256: str,
    implementation_sha256: str,
    initial_checkpoint_sha256: str | None,
    device: torch.device,
    scene_cycle: _PermutationCycle,
    query_cycle: _PermutationCycle,
    action_cycle: _PermutationCycle,
    record: dict[str, object],
    checkpoint_interval: int,
    status_interval: int,
) -> None:
    _save_resume(
        output / "resume.pt",
        carrier=carrier,
        write_rule=write_rule,
        optimizer=optimizer,
        optimizer_phase=optimizer_phase,
        phase=phase,
        phase_step=phase_step,
        global_step=global_step,
        stage_steps=stage_steps,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        source_sha256=source_sha256,
        implementation_sha256=implementation_sha256,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        device=device,
        scene_cycle=scene_cycle,
        query_cycle=query_cycle,
        action_cycle=action_cycle,
        last_history_record=record,
    )
    _append_history(output / "history.jsonl", record)
    if global_step % checkpoint_interval == 0:
        _save_checkpoint(
            output / f"checkpoint_{global_step:06d}.pt",
            carrier,
            write_rule,
            phase=f"phase_{phase.lower()}_step_{phase_step}",
            config_sha256=config_sha256,
            index_sha256=index_sha256,
            source_sha256=source_sha256,
            implementation_sha256=implementation_sha256,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
        )
    if global_step % status_interval == 0:
        _write_status(output / "status.json", "running", phase, phase_step, global_step)


def _save_checkpoint(
    path: Path,
    carrier,
    write_rule,
    *,
    phase: str,
    config_sha256: str,
    index_sha256: str,
    source_sha256: str,
    implementation_sha256: str,
    initial_checkpoint_sha256: str | None = None,
) -> str:
    return save_dynamic_checkpoint(
        path,
        carrier,
        write_rule,
        phase=phase,
        provenance={
            "training_mode": "offline-train-only",
            "config_sha256": config_sha256,
            "episode_index_sha256": index_sha256,
            "validated_manifest_source_sha256": source_sha256,
            "implementation_source_sha256": implementation_sha256,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
        },
    )


def _save_resume(
    path: Path,
    *,
    carrier,
    write_rule,
    optimizer,
    optimizer_phase: str,
    phase: str,
    phase_step: int,
    global_step: int,
    stage_steps: Mapping[str, int],
    config_sha256: str,
    index_sha256: str,
    source_sha256: str,
    implementation_sha256: str,
    initial_checkpoint_sha256: str | None,
    device: torch.device,
    scene_cycle: _PermutationCycle,
    query_cycle: _PermutationCycle,
    action_cycle: _PermutationCycle,
    last_history_record: Mapping[str, object] | None,
) -> None:
    payload = {
        "schema_version": RESUME_SCHEMA,
        "config_sha256": config_sha256,
        "episode_index_sha256": index_sha256,
        "validated_manifest_source_sha256": source_sha256,
        "implementation_source_sha256": implementation_sha256,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "device_type": device.type,
        "carrier_state_dict": _cpu_state_dict(carrier),
        "write_rule_state_dict": _cpu_state_dict(write_rule),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_phase": optimizer_phase,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all()
        if device.type == "cuda"
        else None,
        "schedules": {
            "scene": scene_cycle.state_dict(),
            "query": query_cycle.state_dict(),
            "action": action_cycle.state_dict(),
        },
        "phase": phase,
        "phase_step": phase_step,
        "global_step": global_step,
        "stage_steps": dict(stage_steps),
        "last_history_record": None if last_history_record is None else dict(last_history_record),
    }
    _atomic_torch_save(path, payload)


def _load_resume(path: Path, device: torch.device) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"resume requires an existing resume.pt: {path}")
    value = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(value, dict) or value.get("schema_version") != RESUME_SCHEMA:
        raise ValueError("resume payload is not a supported dynamic training resume")
    return value


def _validate_resume(
    resume: Mapping[str, object],
    *,
    config_sha256: str,
    index_sha256: str,
    source_sha256: str,
    implementation_sha256: str,
    initial_checkpoint_sha256: str | None,
    config: DynamicExperimentConfig,
    device: torch.device,
) -> None:
    expected = {
        "config_sha256": config_sha256,
        "episode_index_sha256": index_sha256,
        "validated_manifest_source_sha256": source_sha256,
        "implementation_source_sha256": implementation_sha256,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "device_type": device.type,
    }
    mismatches = [name for name, value in expected.items() if resume.get(name) != value]
    if mismatches:
        raise ValueError(f"resume hashes or device do not match: {', '.join(mismatches)}")
    for name in ("carrier_state_dict", "write_rule_state_dict", "optimizer_state_dict"):
        if not isinstance(resume.get(name), dict):
            raise ValueError(f"resume {name} must be a dictionary")
    if not isinstance(resume.get("torch_cpu_rng_state"), Tensor):
        raise ValueError("resume must contain CPU torch RNG state")
    if device.type == "cuda" and not isinstance(resume.get("torch_cuda_rng_state"), list):
        raise ValueError("CUDA resume must contain CUDA RNG states")
    stage_steps = resume.get("stage_steps")
    if (
        not isinstance(stage_steps, dict)
        or set(stage_steps) != {"A", "B"}
        or any(type(value) is not int or value < 0 for value in stage_steps.values())
    ):
        raise ValueError("resume stage_steps must contain nonnegative A and B counters")
    for name in ("global_step", "phase_step"):
        if type(resume.get(name)) is not int or int(resume[name]) < 0:
            raise ValueError(f"resume {name} must be a nonnegative integer")
    phase = resume.get("phase")
    optimizer_phase = resume.get("optimizer_phase")
    phase_step = int(resume["phase_step"])
    if phase == "A":
        valid = (
            optimizer_phase == "A"
            and stage_steps["A"] == phase_step
            and stage_steps["B"] == 0
            and phase_step <= config.stage_a_steps
        )
    elif phase == "B":
        valid = (
            optimizer_phase == "B"
            and stage_steps["A"] == config.stage_a_steps
            and stage_steps["B"] == phase_step
            and phase_step <= config.stage_b_steps
        )
    elif phase == "complete":
        valid = (
            optimizer_phase == "B"
            and stage_steps["A"] == config.stage_a_steps
            and stage_steps["B"] == config.stage_b_steps
            and phase_step == config.stage_b_steps
        )
    else:
        valid = False
    if not valid or int(resume["global_step"]) != stage_steps["A"] + stage_steps["B"]:
        raise ValueError("resume phase, optimizer phase, and stage counters are inconsistent")


def _restore_modules(
    config: DynamicExperimentConfig, resume: Mapping[str, object], device: torch.device
) -> tuple[DynamicSceneCarrier, DirectWriteRule]:
    carrier = DynamicSceneCarrier(config.carrier).to(device)
    write_rule = DirectWriteRule(config.carrier, config.write).to(device)
    carrier.load_state_dict(_mapping(resume["carrier_state_dict"]), strict=True)
    write_rule.load_state_dict(_mapping(resume["write_rule_state_dict"]), strict=True)
    return carrier, write_rule


def _restore_global_rng(resume: Mapping[str, object], device: torch.device) -> None:
    torch.set_rng_state(_tensor(resume["torch_cpu_rng_state"], "torch_cpu_rng_state").cpu())
    if device.type == "cuda":
        states = resume["torch_cuda_rng_state"]
        if not isinstance(states, list) or not all(isinstance(state, Tensor) for state in states):
            raise ValueError("resume torch_cuda_rng_state is invalid")
        torch.cuda.set_rng_state_all(states)


def _prepare_output(output: Path, *, resume: bool) -> None:
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {output}")
        return
    if output.exists():
        raise FileExistsError("a new dynamic experiment requires an exclusive output directory")
    output.mkdir(parents=True)


def _write_manifest_provenance(path: Path, entries: Sequence[PilotEntryProvenance]) -> None:
    temporary = path.with_name(f"{path.name}.part")
    write_pilot_provenance(temporary, entries)
    os.replace(temporary, path)


def _append_history(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reconcile_history(
    path: Path, global_step: int, last_history_record: object
) -> None:
    if not path.exists():
        if global_step == 0:
            return
        if not isinstance(last_history_record, Mapping):
            raise ValueError("resume is ahead of a missing history without its last record")
        _append_history(path, last_history_record)
        return
    last_step = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict) or type(value.get("global_step")) is not int:
            raise ValueError(f"history line {line_number} is invalid")
        if value["global_step"] != last_step + 1:
            raise ValueError("history global_step values must be append-only and contiguous")
        last_step = value["global_step"]
    if last_step > global_step:
        raise ValueError("history is ahead of resume state")
    if last_step == global_step:
        return
    if last_step + 1 != global_step or not isinstance(last_history_record, Mapping):
        raise ValueError("history and resume diverge by more than one durable step")
    if last_history_record.get("global_step") != global_step:
        raise ValueError("resume last history record does not match global_step")
    _append_history(path, last_history_record)


def _write_status(path: Path, status: str, phase: str, phase_step: int, global_step: int) -> None:
    _atomic_json(
        path,
        {
            "schema_version": EXPERIMENT_SCHEMA,
            "status": status,
            "phase": phase,
            "phase_step": phase_step,
            "global_step": global_step,
        },
    )


def _completed_report(output: Path) -> dict[str, object]:
    status_path = output / "status.json"
    if not status_path.is_file():
        raise RuntimeError("dynamic experiment completed without a status artifact")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        return {"output_dir": str(output), "status": status}
    return {
        "output_dir": str(output),
        "status": "complete",
        "global_step": status["global_step"],
        "initial_checkpoint": str(output / "initial.pt"),
        "phase_a_final": str(output / "phase_a_final.pt"),
        "phase_b_final": str(output / "phase_b_final.pt"),
    }


def _config_payload(config: DynamicExperimentConfig) -> dict[str, object]:
    return {"schema_version": EXPERIMENT_SCHEMA, "config": asdict(config)}


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _atomic_json(path: Path, value: object) -> None:
    data = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.part")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_checkpoint_hash(
    init_checkpoint: str | Path | None, *, output: Path, resume: bool
) -> str | None:
    if init_checkpoint is None:
        return None
    if resume:
        raise ValueError("--init-checkpoint cannot be combined with --resume")
    checkpoint = Path(init_checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"initial checkpoint does not exist: {checkpoint}")
    if checkpoint == (output / "initial.pt").resolve():
        raise ValueError("initial checkpoint cannot be the output initial.pt artifact")
    return _sha256_file(checkpoint)


def _implementation_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in _IMPLEMENTATION_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"dynamic experiment implementation source is missing: {path}")
        result[str(path.relative_to(_PROJECT)).replace("\\", "/")] = _sha256_file(path)
    return result


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping")
    return value


def _tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    return value
