"""Rebuild stale downloader status from verified existing files, without network I/O."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mcss.data.dl3dv_download import (
    REPO_ID,
    REPO_TYPE,
    REVISION,
    ExclusiveLock,
    atomic_write_json,
    load_inventory_artifacts,
    protocol_frame_selection,
    select_files,
    verify_local_file,
)

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    root = PROJECT / "outputs/dl3dv_download_20260920"
    data = PROJECT / "data/dl3dv_benchmark"
    split = PROJECT / (
        "outputs/benchmark_protocol_20260920/sources/Long-LRM/data/"
        "dl3dv_fold_8_kmeans_input_idx.json"
    )
    if hashlib.sha256(split.read_bytes()).hexdigest() != (
        "4b052aef605fda183683175de66250aaaf337e50003131dbf0113ea781a3145a"
    ):
        raise ValueError("pinned split changed")
    with ExclusiveLock(root / "download.lock"):
        inventory = load_inventory_artifacts(root)
        frames = protocol_frame_selection(split, inventory, data_root=data, target_every=8)
        selected = select_files(
            inventory, data, include_images4=True, frame_ids_by_scene=frames,
        )
        if len(selected) != 12777 or sum(item.remote.size for item in selected) != 11313374180:
            raise ValueError("approved selection changed")
        expected = {item.destination.resolve() for item in selected}
        actual = {path.resolve() for path in data.rglob("*") if path.is_file()}
        if actual != expected:
            raise ValueError("data root has missing, extra, or partial files")

        def check(item):
            result = verify_local_file(item.destination, item.remote)
            return {
                "repo_path": item.remote.path, "destination": str(item.destination),
                "expected_bytes": item.remote.size, **result, "status": "reused",
            }

        with ThreadPoolExecutor(max_workers=4) as pool:
            entries = list(pool.map(check, selected))
        status = root / "status.json"
        backup = root / "status.before_reconciliation_20260920.json"
        if backup.exists():
            raise FileExistsError("reconciliation already attempted; preserve its evidence")
        shutil.copyfile(status, backup)
        old = json.loads(backup.read_text(encoding="utf-8"))
        restored = {
            "schema": "mcss.dl3dv.download.v1", "status": "complete",
            "repo_id": REPO_ID, "repo_type": REPO_TYPE, "revision": REVISION,
            "started_unix": old["started_unix"], "updated_unix": time.time(),
            "reserve_bytes": old["reserve_bytes"], "entries": entries,
            "reconciliation": {
                "reason": "downloader status replacement failed with WinError 5",
                "previous_status": str(backup), "network_used": False,
                "verification": "all existing files rechecked with downloader source hashes",
                "independent_verification": "pending separate verifier",
            },
        }
        atomic_write_json(status, restored)
        print(json.dumps({
            "status": "complete", "files": len(entries), "bytes": 11313374180,
            "independent_verification": "pending",
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
