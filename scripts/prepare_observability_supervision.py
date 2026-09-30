"""Preserve prior experiments and prepare fixed context geometry attribution inputs."""
# ruff: noqa: E501 -- keep frozen scientific contract prose intact

import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.small_training import sha, write_json

OLD = Path("outputs/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1")
CAP = Path("outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1")


def prepare(root, protocol):
    root = Path(root)
    if (root / "scene_manifest.json").exists():
        raise FileExistsError("Preserve existing experiment")
    for sub in ("audit", "raw", "figures", "checkpoints"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    sealed = {}
    for reference in (OLD, CAP):
        for folder, manifest in (
            (reference, "artifact_manifest.json"),
            (Path("docs/experiments") / reference.name, "report_manifest.json"),
        ):
            records = json.loads((folder / manifest).read_text())
            for name, digest in records.items():
                path = folder / name
                if sha(path) != digest:
                    raise PermissionError(f"Old artifact changed: {path}")
                sealed[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
            for name in (manifest, "integrity.json"):
                p = folder / name
                sealed[str(p)] = {"sha256": sha(p), "bytes": p.stat().st_size}
    for name in (
        "scene_manifest.json",
        "bounds_lock.json",
        "bounds_contract.json",
        "train_depth_prior.json",
    ):
        shutil.copy2(OLD / name, root / name)
    shutil.copy2(protocol, root / "USER_PROTOCOL.md")
    for name in ("config.json", "preregistration.json", "state_plan.json", "baseline_results.json"):
        shutil.copy2(OLD / name, root / "audit" / ("previous_" + name))
    write_json(
        root / "audit/previous_experiment_seal.json",
        {"status": "PASS", "files": sealed, "media_read": False, "final_holdout_touched": False},
    )
    rows = json.loads((OLD / "raw/context_results.json").read_text())
    crows = json.loads((OLD / "raw/context_context_results.json").read_text())
    orows = json.loads((OLD / "raw/oracle_results.json").read_text())

    def selected(rs):
        return [
            r
            for r in rs
            if r["bounds_mode"] == "FROZEN_GT_FREE_BOUNDS"
            and r["budget"] == 10000
            and r["selection"] == "FIXED_BUDGET"
            and r["method"] == "direct"
        ]

    rows, crows, orows = map(selected, (rows, crows, orows))

    def equal_scene(rs):
        return float(
            np.mean(
                [
                    np.mean([r["depth_absrel"] for r in rs if r["scene_id"] == sid])
                    for sid in sorted({r["scene_id"] for r in rs})
                ]
            )
        )

    values = {
        "context_absrel": equal_scene(crows),
        "query_absrel": equal_scene(rows),
        "oracle_absrel": equal_scene(orows),
    }
    expected = {
        "context_absrel": 0.0836065732152307,
        "query_absrel": 0.3614798166881315,
        "oracle_absrel": 0.14804454409798198,
    }
    assert all(abs(values[k] - v) < 1e-12 for k, v in expected.items())
    write_json(
        root / "audit/historical_baseline_raw_reproduction.json",
        {
            "status": "PASS",
            "values": values,
            "expected": expected,
            "maximum_error": max(abs(values[k] - v) for k, v in expected.items()),
            "meaning": "Exact saved raw reproduction; fresh S0 optimization and render replay must also pass before S1-S3",
            "query_media_read": False,
        },
    )
    write_json(root / "raw/historical_baseline_query.json", rows)
    write_json(root / "raw/historical_baseline_context.json", crows)
    write_json(root / "raw/historical_oracle_query.json", orows)
    (root / "git_commit.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
    )
    (root / "dirty.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"]))
    write_json(
        root / "audit/preparation.json",
        {
            "status": "PASS",
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "old_files_verified": len(sealed),
            "historical_baseline": values,
            "formal_started": False,
            "new_query_observability_not_read": True,
        },
    )
    print(json.dumps({"old_files_verified": len(sealed), "baseline_raw": values}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--protocol", required=True)
    a = p.parse_args()
    prepare(a.root, a.protocol)
