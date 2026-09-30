#!/usr/bin/env python3
"""Freeze the FRESH-V1 qualification of the sealed V5 RGB-D carriers before any fresh forward."""

from __future__ import annotations

import argparse
import datetime
import shutil
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read, validate_split
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.unseen_data import csv_rows

EXPERIMENT = "EXP-3D-RGBD-FRESH-QUALIFICATION-V1"
V5 = Path("outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5")
DATA = Path("outputs/EXP-3D-RGBD-FRESH-SCALE-DATA-V1")
PARTITIONS = Path("configs/hypersim_er_partitions.csv")


def independence_audit(records, v5_manifest, partitions, protected):
    """Every FRESH record must pass every check; returns the per-scene audit rows."""
    v5_ids = {r["scene_id"] for r in v5_manifest["scenes"]}
    v5_volumes = {s.split("_")[1] for s in v5_ids}
    v5_physical = {r["physical_scene_id"] for r in v5_manifest["scenes"]}
    rows = {r["scene_name"]: r for r in partitions}
    audit, volumes, physical = [], set(), set()
    for record in records:
        sid = record["scene_id"]
        volume = sid.split("_")[1]
        row = rows.get(sid, {})
        checks = {
            "fresh_split_label": record["split"] == "FRESH_QUALIFICATION",
            "official_val_split": row.get("official_split") == "val"
            and row.get("protocol_partition") == "val",
            "never_observed": row.get("previously_observed") == "False",
            "not_protected": sid not in protected,
            "not_in_v5_train_or_dev": sid not in v5_ids,
            "volume_disjoint_from_v5": volume not in v5_volumes,
            "physical_identity_unique": record["physical_scene_id"] not in v5_physical | physical,
            "one_scene_per_volume": volume not in volumes,
            "not_marked_exposed": record.get("historically_exposed") is False,
        }
        volumes.add(volume)
        physical.add(record["physical_scene_id"])
        audit.append({"scene_id": sid, **checks, "verified": all(checks.values())})
    if not audit or not all(a["verified"] for a in audit):
        raise PermissionError(f"FRESH independence audit failed: {audit}")
    return audit


def prepare(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact fresh-qualification directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen protocol text required before freezing")
    v5 = V5.resolve()
    if read(v5 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V5 must be finalized")
    gates = read(v5 / "qualification_results.json")
    if gates["SURFACE_TRAINING_STATUS"] != "SUPPORTED" or gates["STATIC_DEV_STATUS"] != "SUPPORTED":
        raise PermissionError("V5 DEV gates must be SUPPORTED")
    v5_manifest = read(v5 / "scene_split.json")
    data = DATA.resolve()
    fresh = read(data / "manifest_fresh_amended.json")
    protected = set(v5_manifest["protected_scene_ids"]) | set(
        read(data / "candidate_lock.json")["protected_scene_ids"]
    )
    audit = independence_audit(fresh["scenes"], v5_manifest, csv_rows(PARTITIONS), protected)
    scenes = [{**r, "independence_verified": True} for r in fresh["scenes"]]
    manifest = {
        "schema": "mcss.rgbd_fresh_qualification.split.v1",
        "image_size": fresh["image_size"],
        "depth_semantics": "ray_distance_meters",
        "scenes": scenes,
        "protected_scene_ids": sorted(protected),
        "fresh_status": "VERIFIED",
        "FRESH_QUALIFICATION_STATUS": "VERIFIED_INDEPENDENCE_AUDITED",
        "source_manifest_sha256": sha(data / "manifest_fresh_amended.json"),
    }
    validate_split(manifest)
    write_json(root / "scene_split.json", manifest)
    write_json(root / "audit/independence_audit.json", audit)
    shutil.copyfile(v5 / "train_depth_prior.json", root / "train_depth_prior.json")
    v5_contract = read(v5 / "training_contract.json")
    evaluated = read(v5 / "evaluated_variants.json")["variants"]
    contract = {
        "experiment": EXPERIMENT,
        "v5_root": str(v5),
        "seeds": v5_contract["seeds"],
        "variants": evaluated,
        "primary_pair": ["C0", "C1"],
        "num_threads": 1,
        "device": "cpu",
        "fresh_scene_count": len(scenes),
        "fresh_scene_ids": [r["scene_id"] for r in scenes],
        "training_run": False,
        "checkpoints": "the sealed V5 per-seed DEV selections, unchanged",
    }
    write_json(root / "qualification_contract.json", contract)
    v5_rules = read(v5 / "decision_rules.json")
    write_json(
        root / "decision_rules.json",
        {
            "bootstrap": {"unit": "physical scene", "draws": 10000, "seed": 20260928},
            "roster": "FRESH-V1: every audited scene, one per volume; no model or design "
            "decision has used any of them",
            "DEPTH_STATUS_FRESH": "frozen V2 primary-contrast gate on C0-C1 over FRESH-V1 "
            "(the V5 DEPTH_STATUS definition, unchanged)",
            "SCENE_SPECIFICITY_STATUS_FRESH": "C1 wrong-scene damage CIlo>0 and C1 spatial-shuffle "
            "damage CIlo>0 over FRESH-V1",
            "STATIC_STATUS_FRESH": "frozen V2 static gate for C1 over FRESH-V1",
            "FRESH_QUALIFICATION_STATUS": {
                "QUALIFIED": "DEPTH, SCENE_SPECIFICITY and STATIC all SUPPORTED on FRESH-V1",
                "NOT_QUALIFIED": "otherwise",
            },
            "power_note": f"{len(scenes)} scenes, below the data lock's minimum of 6 (Amendment 1 "
            "found no replacement): QUALIFIED still needs every frozen gate; NOT_QUALIFIED may "
            "reflect low power and is reported as such",
            "GEOMETRY_FREE_REFERENCE": v5_rules["GEOMETRY_FREE_REFERENCE"],
            "consequence": "QUALIFIED -> Core A static carrier qualified on the RGB-D track; "
            "DYNAMIC_TTT_NEXT_STAGE_ALLOWED=true under the project rule; the protected official "
            "final holdout stays closed either way",
        },
    )
    names = (
        "scene_split.json",
        "qualification_contract.json",
        "decision_rules.json",
        "train_depth_prior.json",
        "PROTOCOL.md",
        "audit/independence_audit.json",
    )
    selected = read(v5 / "selected_checkpoints.json")
    v5_files = [
        v5 / n
        for n in (
            "preregistration.json",
            "selected_checkpoints.json",
            "evaluated_variants.json",
            "qualification_results.json",
            "decision_rules.json",
            "integrity.json",
            "training_contract.json",
            "scene_split.json",
        )
    ]
    for variant in evaluated:
        run_files = ("summary.json", "training.jsonl", "dev_curve.json")
        for seed in v5_contract["seeds"]:
            v5_files.append(Path(selected[variant][str(seed)]["path"]))
            v5_files += [v5 / "checkpoints" / f"{variant}_{seed}" / n for n in run_files]
    sources = sorted(Path("src/mcss").rglob("*.py"))
    sources += sorted(Path("scripts").glob("*fresh_qualification*.py"))
    sources += sorted(Path("scripts").glob("*rgbd_evidence*.py"))
    data_files = {}
    for record in scenes:
        for frame in record["frames"]:
            for kind in ("rgb", "depth"):
                data_files[str(Path(frame[kind]))] = sha(Path(frame[kind]))
    write_json(
        root / "preregistration.json",
        {
            "status": "FROZEN_BEFORE_FRESH_QUALIFICATION",
            "experiment": EXPERIMENT,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "v5_root": str(v5),
            "fresh_data_root": str(data),
            "input_sha256": {str((root / n).resolve()): sha(root / n) for n in names},
            "source_sha256": {str(p.resolve()): sha(p) for p in sources},
            "data_sha256": data_files,
            "v5_sha256": {str(p.resolve()): sha(p) for p in v5_files},
            "model_forward_on_fresh_before_lock": False,
            "fresh_media_read_before_lock": "data preparation validity checks only (no model)",
            "final_holdout_opened": False,
        },
    )
    print("FROZEN_FRESH_QUALIFICATION", len(scenes), [r["scene_id"] for r in scenes], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    prepare(parser.parse_args().root)
