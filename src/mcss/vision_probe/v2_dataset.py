"""Bounded VOST ZIP range access; original-video provenance is a prerequisite.

A clip name is never promoted to an original-video identity. The default archive
is 54 GB; a server ignoring Range is rejected without consuming its response body.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import requests

from mcss.vision_probe.v2_access import AccessDenied, validate_independent_manifest

VOST_URL = "https://tri-ml-public.s3.amazonaws.com/datasets/VOST.zip"
VOST_SIZE = 54012104924
VOST_ETAG = '"656e2cc81e3dece60378b161992b3f1d-6439"'
BYTE_BUDGET = 1_000_000_000
MAX_RANGE = 25_000_000


class RangeZip(io.RawIOBase):
    """Seekable HTTP range reader with hard byte limits and immutable ETag."""

    def __init__(self, *, budget=BYTE_BUDGET):
        super().__init__()
        if budget <= 0 or budget > BYTE_BUDGET:
            raise ValueError("Range budget must be positive and at most one GB")
        self.position = 0
        self.budget = budget
        self.downloaded_bytes = 0
        self.requests_log = []
        self.session = requests.Session()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else VOST_SIZE)
        if not 0 <= position <= VOST_SIZE:
            raise ValueError("ZIP seek out of range")
        self.position = position
        return position

    def read(self, size=-1):
        length = VOST_SIZE - self.position if size < 0 else min(size, VOST_SIZE - self.position)
        if length == 0:
            return b""
        if length > MAX_RANGE or self.downloaded_bytes + length > self.budget:
            raise RuntimeError("Bounded range/download budget exceeded")
        start, end = self.position, self.position + length - 1
        with self.session.get(
            VOST_URL,
            headers={"Range": f"bytes={start}-{end}", "If-Match": VOST_ETAG},
            stream=True,
            timeout=60,
        ) as response:
            expected = f"bytes {start}-{end}/{VOST_SIZE}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Server did not honor exact bounded byte range")
            if response.headers.get("ETag") != VOST_ETAG:
                raise RuntimeError("VOST archive changed; metadata lock is invalid")
            data = bytearray()
            for chunk in response.iter_content(65536):
                self.downloaded_bytes += len(chunk)
                if len(data) + len(chunk) > length or self.downloaded_bytes > self.budget:
                    raise RuntimeError("Response exceeded byte budget")
                data.extend(chunk)
            if len(data) != length:
                raise RuntimeError("Truncated range response")
        self.position += length
        self.requests_log.append({"start": start, "end": end, "bytes": length})
        return bytes(data)

    def close(self):
        self.session.close()
        super().close()


def select_metadata(index, train_ids, original_video_map, *, seed=20260926, count=32):
    """Choose one clip per proven original video before reading any RGB or GT.

    Mapping entries must contain a sourced original_video_id and evidence_url.
    This function deliberately cannot invent these from VOST's numeric clip IDs.
    """
    if not original_video_map:
        raise ValueError("UNVERIFIED_ORIGINAL_VIDEO_ID: official clip-to-video mapping required")
    names = {row["name"] for row in index}
    by_video = {}
    for clip in train_ids:
        item = original_video_map.get(clip, {})
        if item.get("status") != "VERIFIED" or not item.get("evidence_url"):
            raise ValueError(f"UNVERIFIED original video provenance for {clip}")
        video = item.get("original_video_id")
        if not video:
            raise ValueError(f"Missing original video ID: {clip}")
        by_video.setdefault(video, []).append(clip)

    def key(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    chosen = []
    for video in sorted(by_video, key=key)[:count]:
        clip = sorted(by_video[video], key=key)[0]
        prefix = f"VOST/JPEGImages/{clip}/"
        frames = sorted(name for name in names if name.startswith(prefix) and name.endswith(".jpg"))
        if len(frames) < 3:
            raise ValueError("Locked frame rule requires at least three RGB frames")
        selected = [frames[0], frames[len(frames) // 3], frames[2 * len(frames) // 3]]
        masks = [
            name.replace("/JPEGImages/", "/Annotations/").rsplit(".", 1)[0] + ".png"
            for name in selected
        ]
        if any(name not in names for name in masks):
            raise ValueError("Locked sample lacks annotation-availability metadata; no resampling")
        chosen.append(
            {"sequence": clip, "source_video_id": video, "images": selected, "masks": masks}
        )
    return {
        "seed": seed,
        "n_sequences": len(chosen),
        "n_pairs": 2 * len(chosen),
        "selection_uses": "original video identity, frame index and annotation availability only",
        "sequences": chosen,
    }


# Neither audited archive has passed both source-video and mask-encoding gates.
# An adapter must be separately qualified and tested before this set is extended.
QUALIFIED_TARGET_SCHEMAS: frozenset[str] = frozenset()
SUPPORTS_REMOTE_TARGET_FETCH = False


def fetch_target_masks(manifest, predictions_manifest_path, expected_predictions_sha256, guard):
    """Fail closed until a dataset's identity AND mask encoding are qualified.

    Stable integration API for the independent evaluator. Current VOST and FBMS
    audits are blocked, so this function deliberately performs zero network/file
    writes for every current manifest. It is not a generic image downloader.
    """
    if manifest.get("schema") not in QUALIFIED_TARGET_SCHEMAS:
        raise AccessDenied(
            "NO_QUALIFIED_DATASET_ADAPTER: target schema/identity/mask semantics unverified"
        )
    validate_independent_manifest(manifest)
    if guard.mode != "confirmation" or guard.phase != "evaluation":
        raise AccessDenied("Target downloads require complete independent prediction lock")
    raise AccessDenied("No qualified remote target-mask decoder has been implemented")


def write_metadata_audit(index_path, train_path, readme_path, output_dir):
    """Archive observed provenance limits without pretending clips are independent."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(index_path).read_text())
    train = Path(train_path).read_text().splitlines()
    audit = {
        "dataset": "VOST",
        "official_page": "https://www.vostdataset.org/data.html",
        "archive_url": VOST_URL,
        "archive_size": VOST_SIZE,
        "archive_etag": VOST_ETAG,
        "license": "CC BY-NC-SA 4.0",
        "upstream_sources": ["Ego4D", "EPIC-KITCHENS"],
        "evaluation_domain": "cross-dataset egocentric object transformations",
        "independence_status": "UNVERIFIED",
        "confirmation_allowed": False,
        "reason": "UNVERIFIED_ORIGINAL_VIDEO_ID: release contains clip IDs but no sourced mapping "
        "to Ego4D canonical video IDs or EPIC original video IDs",
        "train_clip_count": len(train),
        "archive_members": len(index),
        "metadata_members": [
            r["name"]
            for r in index
            if r["name"].startswith("VOST/") and r["name"].endswith((".txt", ".md"))
        ],
        "train_identity_examples": train[:16],
        "selected_sequences": [],
        "rgb_downloaded": 0,
        "source_masks_downloaded": 0,
        "target_masks_downloaded": 0,
        "target_gt_read": 0,
        "exact_duplicate_audit": "NOT_RUN_NO_IDENTITY_QUALIFIED_SELECTION",
        "near_duplicate_status": "UNVERIFIED",
        "reserve_touched": False,
        "davis_official_val_touched": False,
        "index_sha256": hashlib.sha256(Path(index_path).read_bytes()).hexdigest(),
        "train_metadata_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        "readme_sha256": hashlib.sha256(Path(readme_path).read_bytes()).hexdigest(),
    }
    (output / "vost_metadata_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "vost_train_identity_metadata.txt").write_text("\n".join(train) + "\n")
    (output / "vost_release_readme.md").write_bytes(Path(readme_path).read_bytes())
    return audit
