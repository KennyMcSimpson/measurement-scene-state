"""Pilot-only manifest provenance checks performed before any image or label decoding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_PREPARED_ROOT = PROJECT / "data" / "hypersim_er_prepared"
ONLINE_EPISODE_SCHEMA = "mcss.dynamic.online_episode.v1"
QUERY_VAULT_SCHEMA = "mcss.dynamic.query_vault.v1"
_ALLOWED_SPLITS = frozenset({"train", "dev"})
# The pilot's logical development split is materialized in Hypersim's val directory.
_PREPARED_SPLIT_BY_INDEX_SPLIT = {"train": "train", "dev": "val"}
_IDENTITY_FIELDS = ("episode_id", "scene_id", "split_id", "query_vault_id")
_INDEX_FIELDS = frozenset(
    {
        *_IDENTITY_FIELDS,
        "online_manifest",
        "query_manifest",
        "stream_steps",
        "query_count",
    }
)
_ONLINE_FIELDS = frozenset(
    {
        "schema_version",
        *_IDENTITY_FIELDS,
        "image_size",
        "declared_length",
        "warmup",
        "stream",
    }
)
_QUERY_FIELDS = frozenset({"schema_version", *_IDENTITY_FIELDS, "image_size", "query"})
_ONLINE_FRAME_FIELDS = frozenset({"frame_id", "intrinsics", "c2w", "rgb"})
_QUERY_FRAME_FIELDS = frozenset({* _ONLINE_FRAME_FIELDS, "depth"})


@dataclass(frozen=True)
class PilotEntryProvenance:
    """Validated metadata and manifest digests suitable for one CLI provenance record."""

    episode_id: str
    scene_id: str
    split_id: str
    query_vault_id: str
    online_manifest: str
    query_manifest: str
    online_manifest_sha256: str
    query_manifest_sha256: str
    online_rgb_references: int
    query_rgb_references: int
    query_depth_references: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_pilot_entry(
    entry: Mapping[str, Any], *, prepared_root: str | Path = DEFAULT_PREPARED_ROOT
) -> PilotEntryProvenance:
    """Validate one compiled pilot entry without opening any raw RGB or depth file."""

    normalized_entry = _validate_index_entry(entry)
    root = Path(prepared_root).resolve()
    online_path = _manifest_path(normalized_entry["online_manifest"], "online_manifest")
    query_path = _manifest_path(normalized_entry["query_manifest"], "query_manifest")
    online = _read_manifest(online_path, "online")
    query = _read_manifest(query_path, "query")
    _validate_manifest_identity(online, normalized_entry, "online")
    _validate_manifest_identity(query, normalized_entry, "query")
    if online["image_size"] != query["image_size"]:
        raise ValueError("online/query image_size metadata mismatch")

    online_records = _online_records(online)
    query_records = _query_records(query)
    if online["declared_length"] != len(online_records):
        raise ValueError("online declared_length does not match warmup plus stream records")
    if normalized_entry["stream_steps"] != len(online["stream"]):
        raise ValueError("index stream_steps does not match online manifest")
    if normalized_entry["query_count"] != len(query_records):
        raise ValueError("index query_count does not match query manifest")

    online_refs = _validate_online_references(online_records, normalized_entry, root)
    query_rgb_refs, query_depth_refs = _validate_query_references(
        query_records, normalized_entry, root
    )
    if set(online_refs) & (set(query_rgb_refs) | set(query_depth_refs)):
        raise ValueError("online and query reference collections must be disjoint")
    return PilotEntryProvenance(
        episode_id=normalized_entry["episode_id"],
        scene_id=normalized_entry["scene_id"],
        split_id=normalized_entry["split_id"],
        query_vault_id=normalized_entry["query_vault_id"],
        online_manifest=str(online_path),
        query_manifest=str(query_path),
        online_manifest_sha256=_sha256(online_path),
        query_manifest_sha256=_sha256(query_path),
        online_rgb_references=len(online_refs),
        query_rgb_references=len(query_rgb_refs),
        query_depth_references=len(query_depth_refs),
    )


def validate_pilot_entries(
    entries: Sequence[Mapping[str, Any]], *, prepared_root: str | Path = DEFAULT_PREPARED_ROOT
) -> tuple[PilotEntryProvenance, ...]:
    """Validate every selected entry before a pilot CLI can open raw observations."""

    if not entries:
        raise ValueError("pilot requires at least one selected episode entry")
    provenances = tuple(
        validate_pilot_entry(entry, prepared_root=prepared_root) for entry in entries
    )
    episode_ids = [item.episode_id for item in provenances]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("pilot index contains duplicate episode_id values")
    return provenances


def write_pilot_provenance(path: str | Path, entries: Sequence[PilotEntryProvenance]) -> Path:
    """Write only validated manifest provenance, after a CLI has completed its work."""

    output = Path(path)
    output.write_text(
        json.dumps(
            {
                "schema_version": "mcss.dynamic.pilot_provenance.v1",
                "entries": [entry.to_dict() for entry in entries],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _validate_index_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError("pilot index entry must be an object")
    _require_exact_fields(entry, _INDEX_FIELDS, "pilot index entry")
    result = dict(entry)
    for name in _IDENTITY_FIELDS:
        _require_identifier(result[name], name)
    if result["split_id"] not in _ALLOWED_SPLITS:
        raise ValueError("pilot split_id must be train or dev")
    for name in ("online_manifest", "query_manifest"):
        if not isinstance(result[name], str) or not result[name]:
            raise ValueError(f"pilot {name} must be a non-empty path string")
    for name in ("stream_steps", "query_count"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"pilot {name} must be a positive integer")
    return result


def _manifest_path(value: str, name: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pilot {name} does not exist: {path}")
    return path


def _read_manifest(path: Path, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {kind} manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{kind} manifest root must be an object")
    expected = _ONLINE_FIELDS if kind == "online" else _QUERY_FIELDS
    schema = ONLINE_EPISODE_SCHEMA if kind == "online" else QUERY_VAULT_SCHEMA
    _require_exact_fields(value, expected, f"{kind} manifest")
    if value["schema_version"] != schema:
        raise ValueError(f"unexpected {kind} manifest schema")
    return value


def _validate_manifest_identity(
    manifest: Mapping[str, Any], entry: Mapping[str, Any], kind: str
) -> None:
    for name in _IDENTITY_FIELDS:
        if manifest[name] != entry[name]:
            raise ValueError(f"{kind} manifest identity mismatch for {name}")


def _online_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    warmup = _validate_frame_records(manifest["warmup"], _ONLINE_FRAME_FIELDS, "online warmup")
    stream = _validate_frame_records(manifest["stream"], _ONLINE_FRAME_FIELDS, "online stream")
    if not warmup or not stream:
        raise ValueError("online manifest requires non-empty warmup and stream records")
    records = [*warmup, *stream]
    _validate_frame_order(records, "online")
    return records


def _query_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = _validate_frame_records(manifest["query"], _QUERY_FRAME_FIELDS, "query")
    if not records:
        raise ValueError("query manifest requires at least one query record")
    _validate_frame_order(records, "query")
    return records


def _validate_frame_records(
    value: Any, expected_fields: frozenset[str], name: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} records must be a list")
    records: list[dict[str, Any]] = []
    for record in value:
        if not isinstance(record, dict):
            raise ValueError(f"{name} records must be objects")
        _require_exact_fields(record, expected_fields, f"{name} frame")
        if type(record["frame_id"]) is not int or record["frame_id"] < 0:
            raise ValueError(f"{name} frame_id must be a non-negative integer")
        records.append(record)
    return records


def _validate_frame_order(records: Sequence[Mapping[str, Any]], name: str) -> None:
    frame_ids = [record["frame_id"] for record in records]
    if len(frame_ids) != len(set(frame_ids)) or frame_ids != sorted(frame_ids):
        raise ValueError(f"{name} frame_ids must be unique and in ascending order")


def _validate_online_references(
    records: Sequence[Mapping[str, Any]], entry: Mapping[str, Any], root: Path
) -> tuple[Path, ...]:
    references = tuple(
        _validate_reference(
            record["rgb"], entry, root, record["frame_id"], "rgb", ".png", "online"
        )
        for record in records
    )
    if len(references) != len(set(references)):
        raise ValueError("online RGB reference collection contains aliases")
    _assert_under_root(references, root)
    return references


def _validate_query_references(
    records: Sequence[Mapping[str, Any]], entry: Mapping[str, Any], root: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    rgb_references = tuple(
        _validate_reference(
            record["rgb"], entry, root, record["frame_id"], "rgb", ".png", "query"
        )
        for record in records
    )
    depth_references = tuple(
        _validate_reference(
            record["depth"], entry, root, record["frame_id"], "depth", ".npy", "query"
        )
        for record in records
    )
    if len(rgb_references) != len(set(rgb_references)):
        raise ValueError("query RGB reference collection contains aliases")
    if len(depth_references) != len(set(depth_references)):
        raise ValueError("query depth reference collection contains aliases")
    _assert_under_root((*rgb_references, *depth_references), root)
    return rgb_references, depth_references


def _validate_reference(
    value: Any,
    entry: Mapping[str, Any],
    root: Path,
    frame_id: int,
    modality: str,
    suffix: str,
    owner: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} {modality} reference must be a non-empty path string")
    expected = (
        root
        / _PREPARED_SPLIT_BY_INDEX_SPLIT[entry["split_id"]]
        / entry["scene_id"]
        / modality
        / f"{frame_id:06d}{suffix}"
    ).resolve()
    actual = Path(value).resolve()
    if actual != expected:
        raise ValueError(
            f"{owner} {modality} reference must resolve to its approved split/scene/frame path"
        )
    if not actual.is_file():
        raise FileNotFoundError(f"approved {owner} {modality} reference is missing: {actual}")
    return actual


def _assert_under_root(references: Sequence[Path], root: Path) -> None:
    for reference in references:
        try:
            reference.relative_to(root)
        except ValueError as error:
            raise ValueError("approved reference escaped prepared_root") from error


def _require_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or any(char in value for char in "\\/"):
        raise ValueError(f"pilot {name} must be a simple non-empty identifier")


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch; missing={missing}, unexpected={unexpected}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
