"""Download and prepare a small, provenance-tracked DAVIS 2D probe asset set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "vision_2d_probe"
RAW_ROOT = DATA_ROOT / "raw"
WEIGHTS_ROOT = DATA_ROOT / "weights"
EXTRACT_ROOT = DATA_ROOT / "davis2017_trainval_480p"
OUTPUT_ROOT = ROOT / "outputs" / "vision_2d_probe_assets_20260921"

DAVIS_PAGE_URL = "https://davischallenge.org/davis2017/code.html"
DAVIS_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/"
    "DAVIS-2017-trainval-480p.zip"
)
DINO_PAGE_URL = "https://github.com/facebookresearch/dinov2"
DINO_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
    "dinov2_vits14_pretrain.pth"
)

USER_AGENT = "Codex-2d-probe-audit/1.0"
CHUNK_SIZE = 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 10_000_000_000
EXPERIMENT_ALLOCATION_COUNTS = {
    "fit": 24,
    "discovery": 12,
    "validation": 16,
    "reserve": 8,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def response_headers(response: requests.Response) -> dict[str, Any]:
    return {
        "status_code": response.status_code,
        "content_length": response.headers.get("Content-Length"),
        "content_type": response.headers.get("Content-Type"),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "final_url": str(response.url),
    }


def get_remote_headers(url: str) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.head(
            url,
            headers=headers,
            allow_redirects=True,
            timeout=(20, 60),
        )
        response.raise_for_status()
        return response_headers(response)
    except requests.RequestException as exc:
        return {"error": f"HEAD failed: {exc}"}


def download(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = get_remote_headers(url)
    expected_size = None
    if remote.get("content_length"):
        try:
            expected_size = int(remote["content_length"])
        except (TypeError, ValueError):
            expected_size = None

    if destination.exists():
        actual_size = destination.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(
                f"Existing file has unexpected size: {destination} "
                f"({actual_size} != {expected_size})"
            )
        return {
            "url": url,
            "destination": str(destination.relative_to(ROOT)),
            "downloaded": False,
            "size_bytes": actual_size,
            "sha256": sha256_file(destination),
            "remote": remote,
            "local_verified_against": "HTTP Content-Length when available; SHA-256 is local only",
        }

    headers = {"User-Agent": USER_AGENT}
    response = requests.get(
        url,
        headers=headers,
        allow_redirects=True,
        stream=True,
        timeout=(20, 120),
    )
    response.raise_for_status()
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".part",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        actual_size = temporary.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(
                f"Downloaded size mismatch for {url}: {actual_size} != {expected_size}"
            )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "url": url,
        "destination": str(destination.relative_to(ROOT)),
        "downloaded": True,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "remote": response_headers(response),
        "local_verified_against": "HTTP Content-Length when available; SHA-256 is local only",
    }


def is_symlink_member(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def safe_member_path(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or any(part in ("", ".", "..") for part in member.parts):
        raise RuntimeError(f"Unsafe ZIP member path: {member_name!r}")
    if len(member.parts[0]) >= 2 and member.parts[0][1] == ":":
        raise RuntimeError(f"Drive-qualified ZIP member path: {member_name!r}")
    target = (root / Path(*member.parts)).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath([str(root_resolved), str(target)]) != str(root_resolved):
        raise RuntimeError(f"ZIP member escapes extraction root: {member_name!r}")
    return target


def safe_extract(archive: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        return {
            "extracted": False,
            "destination": str(destination.relative_to(ROOT)),
            "reason": "destination already exists",
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=destination.name + ".",
            suffix=".extracting",
            dir=str(destination.parent),
        )
    )
    try:
        with zipfile.ZipFile(archive) as handle:
            infos = handle.infolist()
            total_size = sum(info.file_size for info in infos if not info.is_dir())
            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise RuntimeError(
                    f"ZIP expands beyond safety limit: {total_size} bytes"
                )
            for info in infos:
                if is_symlink_member(info):
                    raise RuntimeError(f"Symlink ZIP member rejected: {info.filename!r}")
                target = safe_member_path(staging, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info, "r") as source, target.open("wb") as sink:
                    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                        sink.write(chunk)
        os.replace(staging, destination)
    except BaseException:
        for child in sorted(staging.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        staging.rmdir()
        raise

    return {
        "extracted": True,
        "destination": str(destination.relative_to(ROOT)),
    }


def find_dataset_root(extracted: Path) -> Path:
    candidates = [extracted]
    candidates.extend(path for path in extracted.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "ImageSets" / "2017").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate DAVIS ImageSets/2017 under {extracted}")


def resolve_media_dir(dataset_root: Path, dirname: str) -> Path:
    candidates = [dataset_root / dirname]
    candidates.extend(sorted(dataset_root.glob(f"{dirname}/*")))
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(child.is_dir() and any(child.glob("*.jpg" if dirname == "JPEGImages" else "*.png"))
               for child in candidate.iterdir()):
            return candidate
    raise RuntimeError(f"Could not locate {dirname} under {dataset_root}")


def read_split(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def stable_sequence_rank(name: str) -> str:
    """Return the deterministic UTF-8 name rank used for experiment allocation."""

    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def build_experiment_allocation(
    train_names: list[str],
    val_names: list[str],
) -> dict[str, Any]:
    """Allocate only official train sequences and seal official val sequences."""

    expected_train = sum(EXPERIMENT_ALLOCATION_COUNTS.values())
    if len(train_names) != expected_train:
        raise RuntimeError(
            "Official train sequence count does not match the frozen allocation: "
            f"{len(train_names)} != {expected_train}"
        )
    if len(val_names) != 30:
        raise RuntimeError(
            f"Official val sequence count changed: {len(val_names)} != 30"
        )
    if len(set(train_names)) != len(train_names):
        raise RuntimeError("Official train split contains duplicate sequence names")
    if len(set(val_names)) != len(val_names):
        raise RuntimeError("Official val split contains duplicate sequence names")
    if set(train_names).intersection(val_names):
        raise RuntimeError("Official train/val splits overlap")

    ordered = sorted(train_names, key=stable_sequence_rank)
    groups: dict[str, dict[str, Any]] = {}
    offset = 0
    for split, count in EXPERIMENT_ALLOCATION_COUNTS.items():
        names = ordered[offset : offset + count]
        groups[split] = {"count": len(names), "sequence_names": names}
        offset += count

    return {
        "source_split": "official train.txt",
        "ordering": "ascending sha256(sequence_name encoded as UTF-8)",
        "counts": dict(EXPERIMENT_ALLOCATION_COUNTS),
        "groups": groups,
        "official_val": {
            "source_split": "official val.txt",
            "sequence_count": len(val_names),
            "sequence_names": val_names,
            "sealed": True,
            "used_for_experiment": False,
        },
    }


def sequence_record(
    dataset_root: Path,
    image_root: Path,
    mask_root: Path,
    name: str,
) -> dict[str, Any]:
    image_dir = image_root / name
    mask_dir = mask_root / name
    images = sorted(image_dir.glob("*.jpg"))
    masks = sorted(mask_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No JPEG frames for sequence {name}")
    if not masks:
        raise RuntimeError(f"No PNG masks for sequence {name}")
    first_image = images[0]
    with Image.open(first_image) as image:
        width, height = image.size
    image_names = {path.stem for path in images}
    mask_names = {path.stem for path in masks}
    missing_masks = sorted(image_names - mask_names)
    return {
        "sequence": name,
        "image_dir": str(image_dir.relative_to(dataset_root)),
        "mask_dir": str(mask_dir.relative_to(dataset_root)),
        "frame_count": len(images),
        "mask_count": len(masks),
        "missing_mask_count": len(missing_masks),
        "first_frame": images[0].name,
        "last_frame": images[-1].name,
        "width": width,
        "height": height,
    }


def build_dataset_manifest(
    archive_record: dict[str, Any],
    extraction_record: dict[str, Any],
) -> dict[str, Any]:
    dataset_root = find_dataset_root(EXTRACT_ROOT)
    image_root = resolve_media_dir(dataset_root, "JPEGImages")
    mask_root = resolve_media_dir(dataset_root, "Annotations")
    split_root = dataset_root / "ImageSets" / "2017"
    split_paths = {
        "train": split_root / "train.txt",
        "val": split_root / "val.txt",
    }
    if not split_paths["train"].exists() or not split_paths["val"].exists():
        raise RuntimeError(f"Missing official train/val split files under {split_root}")

    split_names = {name: read_split(path) for name, path in split_paths.items()}
    overlap = sorted(set(split_names["train"]).intersection(split_names["val"]))
    if overlap:
        raise RuntimeError(f"Official train/val sequence overlap: {overlap[:5]}")

    records = {
        "train": [
            sequence_record(dataset_root, image_root, mask_root, name)
            for name in split_names["train"]
        ],
        "val": [
            {"sequence": name, "sealed": True}
            for name in split_names["val"]
        ],
    }
    total_sequences = len(set(split_names["train"]) | set(split_names["val"]))
    if total_sequences < 40:
        raise RuntimeError(f"Too few independent sequences: {total_sequences}")

    return {
        "dataset": "DAVIS-2017 trainval 480p",
        "official_page": DAVIS_PAGE_URL,
        "archive": archive_record,
        "extraction": extraction_record,
        "dataset_root": str(dataset_root.relative_to(ROOT)),
        "image_root": str(image_root.relative_to(dataset_root)),
        "mask_root": str(mask_root.relative_to(dataset_root)),
        "split_root": str(split_root.relative_to(dataset_root)),
        "split_policy": (
            "official train.txt and val.txt are inventory splits; "
            "experimental allocation is recorded separately"
        ),
        "split_counts": {split: len(names) for split, names in split_names.items()},
        "train_val_overlap": overlap,
        "sequences": records,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Prepare DAVIS without downloading the DINOv2 checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = utc_now()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / "prepare.log"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        print(line, end="")

    log("starting 2D probe asset preparation")
    archive_path = RAW_ROOT / Path(DAVIS_URL).name
    archive_record = download(DAVIS_URL, archive_path)
    log(
        f"DAVIS archive ready: {archive_record['size_bytes']} bytes "
        f"sha256={archive_record['sha256']}"
    )
    extraction_record = safe_extract(archive_path, EXTRACT_ROOT)
    log(f"DAVIS extraction: {extraction_record}")

    weight_record: dict[str, Any] | None = None
    if not args.skip_weights:
        weight_path = WEIGHTS_ROOT / Path(DINO_URL).name
        weight_record = download(DINO_URL, weight_path)
        log(
            f"DINOv2 checkpoint ready: {weight_record['size_bytes']} bytes "
            f"sha256={weight_record['sha256']}"
        )

    dataset_manifest = build_dataset_manifest(archive_record, extraction_record)
    experiment_allocation = build_experiment_allocation(
        [record["sequence"] for record in dataset_manifest["sequences"]["train"]],
        [record["sequence"] for record in dataset_manifest["sequences"]["val"]],
    )
    manifest: dict[str, Any] = {
        "schema": "vision_2d_probe_assets.v1",
        "run_started_at": run_started,
        "run_finished_at": utc_now(),
        "project_root": str(ROOT),
        "provenance": {
            "local_sha256_note": (
                "SHA-256 values were computed locally after download; "
                "no official checksum was asserted."
            ),
            "archive_source": DAVIS_URL,
            "archive_source_page": DAVIS_PAGE_URL,
            "checkpoint_source": DINO_URL if weight_record else None,
            "checkpoint_source_page": DINO_PAGE_URL if weight_record else None,
        },
        "dataset": dataset_manifest,
        "experiment_allocation": experiment_allocation,
        "weights": weight_record,
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    log(
        "completed: "
        f"train={manifest['dataset']['split_counts'].get('train', 0)} "
        f"val={manifest['dataset']['split_counts'].get('val', 0)}"
    )


if __name__ == "__main__":
    main()
