#!/usr/bin/env python3
"""Freeze the FRESH-V2 qualification of the sealed V7 RGB-D carriers before any fresh forward."""

from __future__ import annotations

import argparse
import datetime
import shutil
from collections import Counter
from pathlib import Path

from mcss.mechanism_pilot.geometry_carrier_experiment import read, validate_split
from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.unseen_data import csv_rows

EXPERIMENT = "EXP-3D-RGBD-FRESH-QUALIFICATION-V2"
V7 = Path("outputs/EXP-3D-RGBD-DEPTH-BOUNDS-V7")
DATA = Path("outputs/EXP-3D-RGBD-FRESH-V2-DATA")
FRESH_V1 = Path("outputs/EXP-3D-RGBD-FRESH-SCALE-DATA-V1")
PLAN = Path("outputs/EXP-3D-RGBD-RESOLUTION-V8/PLAN_BEFORE_V7_RESULTS.md")
PLAN_SHA256 = "a228ed812c28181db994b8d535e25cb6b390ff6f0ddd80a739c816b519c33232"
PARTITIONS = Path("configs/hypersim_er_partitions.csv")
PER_VOLUME = 2


def independence_audit(records, v7_manifest, earlier_ids, earlier_physical, partitions, protected):
    """Every FRESH-V2 record must pass every check; returns the per-scene audit rows."""
    v7_ids = {r["scene_id"] for r in v7_manifest["scenes"]}
    v7_volumes = {s.split("_")[1] for s in v7_ids}
    v7_physical = {r["physical_scene_id"] for r in v7_manifest["scenes"]}
    rows = {r["scene_name"]: r for r in partitions}
    audit, per_volume, physical = [], Counter(), set()
    for record in records:
        sid = record["scene_id"]
        volume = sid.split("_")[1]
        row = rows.get(sid, {})
        per_volume[volume] += 1
        checks = {
            "fresh_split_label": record["split"] == "FRESH_QUALIFICATION",
            "official_train_split_never_observed": row.get("official_split") == "train"
            and row.get("protocol_partition") == "train"
            and row.get("previously_observed") == "False",
            "not_protected": sid not in protected,
            "not_in_v7_train_or_dev": sid not in v7_ids,
            "never_used_or_considered_earlier": sid not in earlier_ids,
            "volume_disjoint_from_v7_train_and_dev": volume not in v7_volumes,
            "physical_identity_unique": record["physical_scene_id"]
            not in v7_physical | earlier_physical | physical,
            "at_most_two_per_volume": per_volume[volume] <= PER_VOLUME,
            "not_marked_exposed": record.get("historically_exposed") is False,
        }
        physical.add(record["physical_scene_id"])
        audit.append({"scene_id": sid, **checks, "verified": all(checks.values())})
    if not audit or not all(a["verified"] for a in audit):
        raise PermissionError(f"FRESH-V2 independence audit failed: {audit}")
    return audit


def v7_fields(v7):
    fields = dict(
        line.split("=", 1)
        for line in (v7 / "terminal_summary.txt").read_text().splitlines()
        if "=" in line
    )
    return fields


def prepare(root):
    root = Path(root).resolve()
    if root.name != EXPERIMENT:
        raise PermissionError("Exact FRESH-V2 qualification directory required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("No re-freezing")
    if not (root / "PROTOCOL.md").is_file():
        raise PermissionError("Frozen protocol text required before freezing")
    if sha(PLAN) != PLAN_SHA256:
        raise PermissionError("The pre-V7 plan that routes branch A to FRESH-V2 must be intact")
    v7 = V7.resolve()
    if read(v7 / "integrity.json")["status"] != "PASS":
        raise PermissionError("V7 must be finalized")
    gates = read(v7 / "qualification_results.json")
    if gates["SURFACE_TRAINING_STATUS"] != "SUPPORTED" or gates["STATIC_DEV_STATUS"] != "SUPPORTED":
        raise PermissionError("V7 DEV gates must be SUPPORTED")
    fields = v7_fields(v7)
    if fields["INTERPRETATION_BRANCH"] != "A" or fields["C1_ABOVE_REFERENCE_ALL_SCENES"] != "true":
        raise PermissionError("Only a V7 branch-A result may open FRESH-V2")
    v7_manifest = read(v7 / "scene_split.json")
    data = DATA.resolve()
    if read(data / "preparation_integrity.json")["status"] != "PASS":
        raise PermissionError("FRESH-V2 data preparation must PASS")
    fresh = read(data / "manifest_fresh_v2.json")
    lock = read(data / "candidate_lock.json")
    if fresh["candidate_lock_sha256"] != sha(data / "candidate_lock.json"):
        raise PermissionError("FRESH-V2 manifest does not match its candidate lock")
    earlier = set(lock["used_scene_ids"])
    v1_manifest = read(FRESH_V1 / "manifest_fresh_amended.json")
    earlier_physical = {r["physical_scene_id"] for r in v1_manifest["scenes"]}
    protected = set(v7_manifest["protected_scene_ids"]) | set(lock["protected_scene_ids"])
    audit = independence_audit(
        fresh["scenes"], v7_manifest, earlier, earlier_physical, csv_rows(PARTITIONS), protected
    )
    scenes = [{**r, "independence_verified": True} for r in fresh["scenes"]]
    manifest = {
        "schema": "mcss.rgbd_fresh_qualification.split.v2",
        "image_size": fresh["image_size"],
        "depth_semantics": "ray_distance_meters",
        "scenes": scenes,
        "protected_scene_ids": sorted(protected),
        "fresh_status": "VERIFIED",
        "FRESH_QUALIFICATION_STATUS": "VERIFIED_INDEPENDENCE_AUDITED",
        "source_manifest_sha256": sha(data / "manifest_fresh_v2.json"),
    }
    validate_split(manifest)
    write_json(root / "scene_split.json", manifest)
    write_json(root / "audit/independence_audit.json", audit)
    shutil.copyfile(v7 / "train_depth_prior.json", root / "train_depth_prior.json")
    v7_contract = read(v7 / "training_contract.json")
    evaluated = read(v7 / "evaluated_variants.json")["variants"]
    volumes = Counter(r["scene_id"].split("_")[1] for r in scenes)
    contract = {
        "experiment": EXPERIMENT,
        "v7_root": str(v7),
        "seeds": v7_contract["seeds"],
        "variants": evaluated,
        "primary_pair": ["C0", "C1"],
        "num_threads": 1,
        "device": "cpu",
        "fresh_scene_count": len(scenes),
        "fresh_scene_ids": [r["scene_id"] for r in scenes],
        "fresh_volumes": dict(sorted(volumes.items())),
        "training_run": False,
        "checkpoints": "the sealed V7 per-seed DEV selections, unchanged",
        "bounds": "each V7 variant keeps its own rule; context-depth bounds from FRESH context "
        "depth only, before any state is sealed",
        "estimator": "V7 Amendment 1: the frozen V2 estimator without its inapplicable C0/C1 "
        "ray-hit equality assertion",
    }
    write_json(root / "qualification_contract.json", contract)
    v7_rules = read(v7 / "decision_rules.json")
    write_json(
        root / "decision_rules.json",
        {
            "bootstrap": {"unit": "physical scene", "draws": 10000, "seed": 20260928},
            "roster": "FRESH-V2: every audited scene (<=2 per volume); no model or design "
            "decision has used any of them",
            "BOUNDS_STATUS_FRESH": "frozen V2 primary-contrast gate on C0-C1 over FRESH-V2 (the "
            "V7 BOUNDS_STATUS definition under V7 Amendment 1, unchanged)",
            "SCENE_SPECIFICITY_STATUS_FRESH": "C1 wrong-scene damage CIlo>0 and C1 "
            "spatial-shuffle damage CIlo>0 over FRESH-V2",
            "STATIC_STATUS_FRESH": "frozen V2 static gate for C1 over FRESH-V2",
            "REFERENCE_STATUS_FRESH": "C1 RAW depth AbsRel ABOVE REF_TRAIN_ABSREL_OPTIMAL on ALL "
            "FRESH-V2 scenes (frozen V2 evidence rule: mean>0, CIlo>0, >=75% scenes non-worse) "
            "-> SUPPORTED, otherwise NOT_ESTABLISHED",
            "FRESH_QUALIFICATION_STATUS": {
                "QUALIFIED": "BOUNDS, SCENE_SPECIFICITY, STATIC and REFERENCE all SUPPORTED on "
                "FRESH-V2",
                "NOT_QUALIFIED": "otherwise",
            },
            "descriptive_only": "volume-cluster bootstrap of BOUNDS_GAIN and of C1 versus the "
            "constant (10 volumes); per-scene table; OBS regions; C2; ray-hit fractions",
            "GEOMETRY_FREE_REFERENCE": v7_rules["GEOMETRY_FREE_REFERENCE"],
            "consequence": "QUALIFIED -> Core A static carrier qualified on the RGB-D track; "
            "DYNAMIC_TTT_NEXT_STAGE_ALLOWED=true under the project rule; the protected official "
            "final holdout stays closed either way",
            "one_look": "FRESH-V2 is evaluated once; no rule, scene, threshold or checkpoint "
            "changes after any FRESH-V2 forward",
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
    selected = read(v7 / "selected_checkpoints.json")
    v7_files = [
        v7 / n
        for n in (
            "preregistration.json",
            "selected_checkpoints.json",
            "evaluated_variants.json",
            "qualification_results.json",
            "decision_rules.json",
            "integrity.json",
            "training_contract.json",
            "scene_split.json",
            "terminal_summary.txt",
            "AMENDMENTS.md",
        )
    ]
    for variant in evaluated:
        run_files = ("summary.json", "training.jsonl", "dev_curve.json")
        for seed in v7_contract["seeds"]:
            v7_files.append(Path(selected[variant][str(seed)]["path"]))
            v7_files += [v7 / "checkpoints" / f"{variant}_{seed}" / n for n in run_files]
    sources = sorted(Path("src/mcss").rglob("*.py"))
    sources += sorted(Path("scripts").glob("*fresh_v2*.py"))
    sources += sorted(Path("scripts").glob("*rgbd_bounds*.py"))
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
            "v7_root": str(v7),
            "fresh_data_root": str(data),
            "input_sha256": {str((root / n).resolve()): sha(root / n) for n in names},
            "source_sha256": {str(p.resolve()): sha(p) for p in sources},
            "data_sha256": data_files,
            "v7_sha256": {str(p.resolve()): sha(p) for p in v7_files},
            "model_forward_on_fresh_before_lock": False,
            "fresh_media_read_before_lock": "data preparation validity checks only (no model)",
            "final_holdout_opened": False,
        },
    )
    print("FROZEN_FRESH_V2_QUALIFICATION", len(scenes), dict(sorted(volumes.items())), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs") / EXPERIMENT)
    prepare(parser.parse_args().root)
