#!/usr/bin/env python3
"""Reconstruct frozen carrier states from exposed contexts; no writer or query reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.checkpoint import _carrier_config
from mcss.dynamic.feedback import anchored_observation
from mcss.dynamic.types import hash_scene_state
from mcss.mechanism_pilot.small_training import TrainScene, sha, write_json

CHECKPOINT = Path(
    "outputs/EXP-3D-20260927-centered-training-1000a-600b-v1/training/phase_b_final.pt"
)
STATES = Path("outputs/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1/raw/oracle_run/sealed_states.pt")
EXPECTED = "be7b8b6d2ef366cad245732b9ff227802db596da65082ac0e160f24541579e42"


@torch.no_grad()
def reconstruct(root, device="cuda"):
    root = Path(root)
    manifest = json.loads((root / "scene_manifest.json").read_text())
    forbidden = set(manifest["FINAL_HOLDOUT_PROHIBITED"])
    assert sha(CHECKPOINT) == EXPECTED
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    assert payload["schema_version"] == "mcss.dynamic.v1"
    # Instantiate and restore carrier only: no DirectWriteRule construction or invocation.
    carrier = DynamicSceneCarrier(_carrier_config(payload["carrier_config"])).to(device)
    carrier.load_state_dict(payload["carrier_state_dict"], strict=True)
    carrier.eval().requires_grad_(False)
    previous = torch.load(STATES, map_location=device, weights_only=False)
    rows, access = [], []
    for scene in manifest["scenes"]:
        sid = scene["scene_id"]
        if sid in forbidden or scene["cohort_role"] != "CAPACITY_DEV_EXPOSED":
            raise PermissionError("Protected/unknown scene")
        # Strip all query frames before creating even an ordinary media loader.
        arrived = set(scene["roles"]["context_a"] + scene["roles"]["context_b"])
        private = dict(scene, frames=[f for f in scene["frames"] if f["frame_id"] in arrived])
        loader = TrainScene(private, manifest["image_size"], root, device, access)
        observations = {
            o.frame_id: o for role in ("context_a", "context_b") for o in loader.observations(role)
        }
        anchor = observations[scene["roles"]["context_a"][0]].camera.c2w
        for role, ids in [
            ("A", scene["roles"]["context_a"]),
            ("B", scene["roles"]["context_b"]),
            ("anchor", scene["roles"]["context_a"][:1]),
        ]:
            key = f"R0/{sid}/{role}"
            episode = key.replace("/", "-")
            cache = ObservationCache(episode, sid)
            fast = carrier.initial_fast(episode)  # Immutable zeros required by carrier read API.
            for fid in ids:
                obs = anchored_observation(observations[fid], anchor)
                cache = cache.append(obs, carrier.encode(obs))
            state = carrier.materialize(cache, fast)
            actual, expected = hash_scene_state(state), previous[key].state_hash
            assert not fast.delta_fuse.any() and not fast.delta_complete.any()
            rows.append(
                {
                    "scene_id": sid,
                    "role": role,
                    "observed_frame_ids": ids,
                    "actual_state_hash": actual,
                    "previous_state_hash": expected,
                    "exact": actual == expected,
                    "query_access": False,
                }
            )
    result = {
        "status": "PASS" if len(rows) == 51 and all(r["exact"] for r in rows) else "FAIL",
        "reconstructed_states": len(rows),
        "rows": rows,
        "prior": (
            "Frozen analytic rgb=0.5 depth=5 opacity=1, already replayed exactly "
            "in136row baseline; no carrier state exists for this prior."
        ),
        "checkpoint_sha256": sha(CHECKPOINT),
        "checkpoint_unchanged": sha(CHECKPOINT) == EXPECTED,
        "carrier_training": False,
        "writer_instantiated": False,
        "writer_invoked": False,
        "query_GT_or_camera_loaded": False,
        "final_holdout_loaded": False,
    }
    write_json(root / "baseline_carrier_reconstruction.json", result)
    write_json(root / "audit/baseline_carrier_context_access.json", access)
    if result["status"] != "PASS":
        raise RuntimeError("Frozen carrier reconstruction mismatch")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    r = reconstruct(a.root, a.device)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2))
