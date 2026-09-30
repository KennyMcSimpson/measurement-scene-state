"""Execute a frozen state plan; context and privileged oracle phases stay separate."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from mcss.geometry import transform_cameras
from mcss.mechanism_pilot.direct_capacity_contracts import (
    ContextOnlyLoader,
    ExplicitQueryOracleLoader,
)
from mcss.mechanism_pilot.direct_capacity_optimization import (
    DirectOptimizationConfig,
    optimize_state,
)
from mcss.mechanism_pilot.small_training import sha, write_json


def run(root, phase, device):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    plan = json.loads((root / "state_plan.json").read_text())
    manifest = json.loads((root / "scene_manifest.json").read_text())
    records = {s["scene_id"]: s for s in manifest["scenes"]}
    ids, held = tuple(records), tuple(manifest["FINAL_HOLDOUT_PROHIBITED"])
    if set(ids) & set(held) or len(ids) != 17:
        raise PermissionError("Only fixed17 capacity-exposed scenes permitted")
    if phase == "oracle":
        marker = root / "raw/context_evaluation_integrity.json"
        if not marker.exists() or json.loads(marker.read_text())["status"] != "PASS":
            raise PermissionError("Privileged oracle must follow sealed context-only evaluation")
    specs = [s for s in plan["states"] if s["phase"] == phase]
    locked = {
        str(root / p): sha(root / p)
        for p in [
            "config.json",
            "state_plan.json",
            "scene_manifest.json",
            "predeclared_geometry_bounds.json",
            "preregistration.json",
        ]
    }
    for p, h in config["source_sha256"].items():
        if sha(Path(p)) != h:
            raise PermissionError("Frozen optimizer/loader implementation changed")
    settings = DirectOptimizationConfig(**config["optimization"])
    for i, spec in enumerate(specs):
        dest = root / spec["optimization_path"]
        # Fail closed on partial work; completed frozen states may be resumed only by hash.
        if dest.exists():
            summary_path = dest / "summary.json"
            if not summary_path.exists() or not (dest / "completion.json").exists():
                raise PermissionError(f"Partial optimization retained; investigate {dest}")
            completed = json.loads((dest / "completion.json").read_text())
            assert completed["plan_sha256"] == sha(root / "state_plan.json")
            assert completed["state_file_sha256"] == sha(dest / "state.pt")
            continue
        scene = records[spec["scene_id"]]
        access = []
        kwargs = {"capacity_scene_ids": ids, "holdout_scene_ids": held, "access_log": access}
        if phase == "context":
            loader = ContextOnlyLoader(
                scene, manifest["image_size"], root, device, track=spec["track"], **kwargs
            )
            batch = loader.context(spec["role"])
            anchor = batch.cameras.c2w[0, 0]
        else:
            loader = ExplicitQueryOracleLoader(
                scene,
                manifest["image_size"],
                root,
                device,
                scope="QUERY_SUPERVISED_ORACLE",
                **kwargs,
            )
            # Anchor is arrived-context camera, not inferred from query geometry.
            arrived = ContextOnlyLoader(
                scene, manifest["image_size"], root, device, track="RGB_ONLY", **kwargs
            ).context("anchor")
            anchor = arrived.cameras.c2w[0, 0]
            batch = loader.diagnostic_query_supervision()
        batch = replace(batch, cameras=transform_cameras(batch.cameras, torch.linalg.inv(anchor)))
        bounds = torch.tensor(spec["bounds"], dtype=torch.float32, device=device)
        seed = 20260927 + int(
            hashlib.sha256(
                f"{spec['scene_id']}|{spec['role']}|{spec['track']}".encode()
            ).hexdigest()[:8],
            16,
        )
        state, summary = optimize_state(
            batch, bounds, spec["grid"], spec["samples"], dest, seed=seed, config=settings
        )
        write_json(dest / "access.json", access)
        write_json(
            dest / "completion.json",
            {
                "plan_sha256": sha(root / "state_plan.json"),
                "state_file_sha256": sha(dest / "state.pt"),
                "spec": spec,
                "phase": phase,
                "source_hashes": config["source_sha256"],
            },
        )
        print(
            f"{phase} {i + 1}/{len(specs)} {spec['key']} "
            f"best={summary['selected_step']} "
            f"objective={summary['selected']['context_objective']:.6f} "
            f"seconds={summary['seconds']:.2f}",
            flush=True,
        )
        del state, batch, loader
    assert all(sha(Path(p)) == h for p, h in locked.items())
    assert all(sha(Path(p)) == h for p, h in config["source_sha256"].items())
    write_json(
        root / f"{phase}_optimization_complete.json",
        {
            "status": "PASS",
            "states": len(specs),
            "phase": phase,
            "input_hashes": locked,
            "final_holdout_touched": False,
            "new_carrier_trained": False,
            "dynamic_ttt_run": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--phase", choices=["context", "oracle"], required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    a = p.parse_args()
    run(a.root, a.phase, a.device)
