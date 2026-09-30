#!/usr/bin/env python3
"""Independently cross-check frozen unseen roles, identities, and allowed media."""

import argparse
import csv
import json
import re
from pathlib import Path

from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.unseen_data import asset_identity, source_assets


def audit(root):
    root = Path(root)
    data = root / "unseen_data"
    candidate = json.loads((data / "candidate_lock.json").read_text())
    member_lock = json.loads((data / "data_lock.json").read_text())
    role = json.loads((data / "role_lock.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    assets = source_assets(root / "metadata/upstream_dataset_config.py")
    train = candidate["train_scene_ids"]
    selected = candidate["candidate_scene_ids"]
    qualified = [r["scene_id"] for r in manifest["scenes"]]
    with Path("configs/hypersim_er_partitions.csv").open() as f:
        partitions = {r["scene_name"]: r for r in csv.DictReader(f)}
    with (root / "rgb_audit/official_metadata/metadata_images_split_scene_v1.csv").open() as f:
        official = {
            (r["scene_name"], r["camera_name"], int(r["frame_id"])): r for r in csv.DictReader(f)
        }
    with (root / "rgb_audit/official_metadata/metadata_camera_trajectories.csv").open() as f:
        trajectories = {r["Animation"]: r for r in csv.DictReader(f)}
    source_hash_mismatch = [p for p, h in candidate["source_sha256"].items() if sha(Path(p)) != h]
    accesses = [json.loads(x) for x in (data / "access.jsonl").read_text().splitlines()]
    downloads = [r for r in accesses if r["operation"] == "download"]
    bad_media, bad_roles = [], []
    for plan in member_lock["scenes"]:
        sid = plan["scene_id"]
        for member in plan["members"]:
            match = re.search(r"/frame\.(\d{4})\.", member)
            if match:
                row = official.get((sid, "cam_00", int(match[1])))
                if (
                    not row
                    or row["included_in_public_release"] != "True"
                    or row["split_partition_name"] != "train"
                ):
                    bad_media.append(member)
    for s in manifest["scenes"]:
        r = s["roles"]
        if r != role["roles"][s["scene_id"]] or set(r["primary_query"]) & (
            set(r["context_a"]) | set(r["context_b"])
        ):
            bad_roles.append(s["scene_id"])
    source_rows = {
        s: {
            "archive_file": assets[s]["archive_file"],
            "asset_file": assets[s]["asset_file"],
            "volume": s.split("_")[1],
        }
        for s in train + selected
    }
    identities = [asset_identity(assets[s]) for s in train + selected]
    first_download = min(r["time"] for r in downloads)
    static_lock = root / "unseen_static/lock.json"
    static_lock_mtime = static_lock.stat().st_mtime if static_lock.exists() else None
    checks = {
        "train_scene_overlap_absent": not set(train) & set(selected),
        "source_asset_pairs_all_distinct": len(identities) == len(set(identities)),
        "training_volumes_disjoint": not {s.split("_")[1] for s in train}
        & {s.split("_")[1] for s in selected},
        "source_hashes_unchanged": not source_hash_mismatch,
        "candidate_lock_hash_matches_data_lock": sha(data / "candidate_lock.json")
        == member_lock["candidate_lock_sha256"],
        "data_lock_hash_matches_manifest": sha(data / "data_lock.json")
        == manifest["data_lock_sha256"],
        "manifest_hash_matches_role_lock": sha(data / "manifest.json") == role["manifest_sha256"],
        "only_locked_candidates_downloaded": {r["scene_id"] for r in downloads} == set(selected),
        "all_candidate_protocol_partitions_train": all(
            partitions[s]["protocol_partition"] == "train" for s in selected
        ),
        "all_candidate_trajectories_non_BAD": all(
            "BAD" not in trajectories[s + "_cam_00"]["Scene type"].upper() for s in selected
        ),
        "all_media_official_included_train": not bad_media,
        "candidate_then_members_then_download_order": candidate["created_unix"]
        < member_lock["created_unix"]
        < first_download,
        "locked_roles_identical_and_queries_disjoint": not bad_roles,
        "role_lock_before_static_lock_local_mtime": static_lock_mtime is not None
        and role["created_unix"] < static_lock_mtime,
        "qualified_scenes_subset_frozen_candidates": set(qualified) <= set(selected),
    }
    report = {
        "status": "PASS_WITH_PHYSICAL_IDENTITY_LIMITATION" if all(checks.values()) else "FAIL",
        "checks": checks,
        "train_scene_ids": train,
        "candidate_scene_ids": selected,
        "qualified_scene_ids": qualified,
        "source_asset_mapping": source_rows,
        "previously_observed": candidate["previously_observed"],
        "independence_scope": (
            "Unseen to this frozen B-final checkpoint; ai_007_002 has prior "
            "project exposure and is not claimed globally virgin."
        ),
        "physical_identity_evidence": (
            "Distinct official archive/asset pairs and volumes; source mappings "
            "immutable by SHA256."
        ),
        "physical_identity_limit": (
            "Vendor content duplication across different source assets cannot be"
            " ruled out; no claim of externally certified physical-location "
            "uniqueness."
        ),
        "source_url": (
            "https://github.com/apple/ml-hypersim/blob/main/evermotion_dataset/_dataset_config.py"
        ),
        "candidate_lock_url_erratum": str(data / "preparation_notes.json"),
        "protected_test_media_opened": False,
        "bad_media": bad_media,
        "bad_roles": bad_roles,
        "source_hash_mismatch": source_hash_mismatch,
        "order_evidence": {
            "candidate_unix": candidate["created_unix"],
            "members_unix": member_lock["created_unix"],
            "first_media_download_unix": first_download,
            "role_lock_unix": role["created_unix"],
            "static_lock_mtime_unix": static_lock_mtime,
            "limitation": (
                "Local operation timestamps and file mtime; not an external "
                "cryptographic timestamp service."
            ),
        },
        "artifact_sha256": {
            str(p): sha(p)
            for p in (
                data / "candidate_lock.json",
                data / "data_lock.json",
                data / "role_lock.json",
                data / "manifest.json",
                data / "access.jsonl",
                root / "metadata/upstream_dataset_config.py",
            )
        },
    }
    write_json(root / "unseen_independence_audit.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.experiment), indent=2))
