"""Compare two frozen geometries using observed cameras/RGB only; no query access."""

import argparse
import json
from pathlib import Path

import torch

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.carrier import DynamicSceneCarrier
from mcss.dynamic.config import WriteConfig
from mcss.dynamic.feedback import anchored_observation
from mcss.dynamic.write_rule import DirectWriteRule
from mcss.mechanism_pilot.small_training import TrainScene, sha, write_json
from mcss.mechanism_pilot.spatial import (
    SPATIAL_MODES,
    carrier_config_for_spatial_mode,
    observed_camera_support,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    access, rows = [], []
    with torch.no_grad():
        for mode in SPATIAL_MODES:
            torch.manual_seed(20260927)
            carrier = DynamicSceneCarrier(carrier_config_for_spatial_mode(mode))
            writer = DirectWriteRule(carrier.config, WriteConfig())
            for record in manifest["scenes"]:
                if record["split"] != "train":
                    raise ValueError("This diagnostic is train-only")
                # Narrow the record before loading media; no query camera or depth is used.
                ids = record["roles"]["context_a"] + record["roles"]["stream"]
                observed = {
                    **record,
                    "frames": [f for f in record["frames"] if f["frame_id"] in ids],
                }
                scene = TrainScene(
                    observed, manifest["image_size"], args.manifest.parent, "cpu", access
                )
                cameras = [scene.camera(scene.frames[i]) for i in ids]
                row = {
                    "mode": mode,
                    "scene_id": record["scene_id"],
                    "frame_ids": ids,
                    **observed_camera_support(carrier._candidate_points, cameras),
                }
                observations = scene.observations("context_a") + scene.observations("stream")
                cache = ObservationCache("observed-support", record["scene_id"])
                fast = carrier.initial_fast(cache.episode_id)
                proposals = []
                for observation in observations:
                    arrived = anchored_observation(observation, observations[0].camera.c2w)
                    cache = cache.append(arrived, carrier.encode(arrived))
                    trace = carrier.trace(cache, fast)
                    proposal = writer.propose(trace)
                    proposals.append(
                        {
                            "frame_id": observation.frame_id,
                            "supported_candidates": int((trace.support_weights > 0).sum()),
                            "fuse_norm": float(proposal.delta_fuse.norm()),
                            "complete_norm": float(proposal.delta_complete.norm()),
                        }
                    )
                assert [p["supported_candidates"] for p in proposals] == row[
                    "supported_candidates_per_prefix"
                ]
                row["zero_fast_proposals_per_prefix"] = proposals
                rows.append(row)
    write_json(
        args.output,
        {
            "scope": "OBSERVED_ONLY_TRAIN_GEOMETRY_DIAGNOSTIC",
            "manifest_sha256": sha(args.manifest),
            "rows": rows,
            "access_log": access,
            "query_media_reads": 0,
            "depth_reads": 0,
            "support_semantics": "At least two camera frusta; not surface visibility",
        },
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
