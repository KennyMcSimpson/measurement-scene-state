"""Supplement missing protocol frames, retaining the original download evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mcss.data.dl3dv_download import (
    REVISION,
    atomic_write_json,
    download_selected,
    load_inventory_artifacts,
    protocol_frame_selection,
    select_files,
    write_inventory_artifacts,
)

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(PROJECT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    old_status_path = args.previous / "status.json"
    old_status = json.loads(old_status_path.read_text(encoding="utf-8"))
    old_verification_path = args.previous / "completion_verification.json"
    old_verification = json.loads(old_verification_path.read_text(encoding="utf-8"))
    if old_status.get("status") != "complete" or old_status.get("revision") != REVISION:
        raise ValueError("previous download must be complete and use the pinned revision")
    if old_verification.get("status") != "verified":
        raise ValueError("previous file-integrity verification is absent")
    inventory = load_inventory_artifacts(args.previous)
    frames = protocol_frame_selection(
        args.protocol_json, inventory, data_root=args.data_root, target_every=8,
    )
    selected = select_files(
        inventory, args.data_root, include_images4=True, frame_ids_by_scene=frames,
    )
    if len(selected) != 12777 or sum(x.remote.size for x in selected) != 11313259491:
        raise ValueError("corrected protocol selection differs from the audited selection")
    old_by_path = {entry["repo_path"]: entry for entry in old_status["entries"]}
    old_verified = {entry["repo_path"]: entry for entry in old_verification["entries"]}
    required = {x.remote.path for x in selected}
    missing = [x for x in selected if not x.destination.is_file()]
    if len(missing) != 281 or sum(x.remote.size for x in missing) != 245981462:
        raise ValueError("missing files differ from the audited 281-file supplement")
    plan = {
        "reason": "protocol frame positions were incorrectly mapped to filename index + 1",
        "revision": REVISION,
        "mapping": "transforms_frame_order",
        "required_file_count": len(selected),
        "required_bytes": sum(x.remote.size for x in selected),
        "missing_file_count": len(missing),
        "missing_bytes": sum(x.remote.size for x in missing),
        "previous_status_sha256": hashlib.sha256(old_status_path.read_bytes()).hexdigest(),
        "previous_verification_sha256": hashlib.sha256(
            old_verification_path.read_bytes()
        ).hexdigest(),
        "retained_unused_repo_paths": sorted(set(old_by_path) - required),
        "missing_repo_paths": [x.remote.path for x in missing],
        "independent_verification": "pending",
    }
    atomic_write_json(output / "repair_plan.json", plan)
    write_inventory_artifacts(inventory, output)
    print(json.dumps({k: v for k, v in plan.items() if not isinstance(v, list)}), flush=True)
    supplement = download_selected(
        missing, status_path=output / "supplement_status.json",
        lock_path=args.previous / "download.lock", workers=4, max_attempts=3,
    )
    downloaded = {entry["repo_path"]: entry for entry in supplement["entries"]}
    entries = []
    for item in selected:
        path = item.remote.path
        if path in downloaded:
            entry = downloaded[path]
        else:
            entry = dict(old_by_path[path])
            if (
                entry.get("status") not in {"reused", "downloaded"}
                or entry.get("sha256") != old_verified[path]["local_sha256"]
            ):
                raise ValueError(f"prior status and prior verifier disagree: {path}")
            entry["status"] = "reused"
        entries.append(entry)
    # This records transfer completion only. The separate verifier re-hashes
    # every required and retained file before any benchmark restart.
    atomic_write_json(output / "status.json", {
        "schema": "mcss.dl3dv.download.v1", "status": "complete",
        "revision": REVISION, "entries": entries, "repair_plan": str(output / "repair_plan.json"),
        "independent_verification": "pending",
    })
    print(json.dumps({"status": "complete", "supplement_files": len(missing)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
