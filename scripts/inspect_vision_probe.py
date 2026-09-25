"""Post hoc resolution and feature-change diagnostics; never selects parameters."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from mcss.vision_probe.adaptation import Readout, materialize  # noqa: E402
from mcss.vision_probe.experiment import (  # noqa: E402
    _cached_extract,
    _run_pair,
    discover_sequences,
)


def inspect(directory, data_root):
    torch.set_num_threads(1)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "complete"
    cache = summary["feature_cache"]
    payload = torch.load(directory / "readout.pt", weights_only=True)
    readout = Readout(*(payload[key] for key in ("mean", "basis", "scale", "decoder", "bias")))
    sequences = {seq.name: seq for seq in discover_sequences(data_root)}
    rows = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
    methods = {"OFFOFF", "bestgloballyfixed", "residual_selector", "oracle_bestsequenceperpair"}
    rows = [row for row in rows if row["split"] == "validation" and row["method"] in methods]
    details = []
    for row in rows:
        sequence = sequences[row["sequence"]]
        frames = sequence.probe_frames()
        source, target = [
            _cached_extract(
                None,
                frame,
                directory / "cache",
                cache["weight_sha256"],
                cache["model_source_sha256"],
                cache["size"],
            )
            for frame in (frames[0], frames[row["target_slot"]])
        ]
        state, _ = _run_pair(source, target, readout, tuple(row["action_pair"]), row["eta"])
        z = readout.project(target.features)
        adapted = readout.restore(target.features, z, materialize(z, state))
        source_flat = F.normalize(source.features.flatten(0, 1), dim=-1)
        original_flat = F.normalize(target.features.flatten(0, 1), dim=-1)
        adapted_flat = F.normalize(adapted.flatten(0, 1), dim=-1)
        old_nn = (original_flat @ source_flat.T).argmax(1)
        new_nn = (adapted_flat @ source_flat.T).argmax(1)
        label_maps = []
        for path in (sequence.probe_masks()[0], sequence.probe_masks()[row["target_slot"]]):
            with Image.open(path) as image:
                label_maps.append(
                    torch.from_numpy(
                        np.asarray(image.resize((16, 16), Image.Resampling.NEAREST)).copy()
                    ).flatten()
                )
        source_labels, target_labels = label_maps
        # DAVIS train masks in this probe have no ignore pixels. Fail rather than
        # silently diverge from the production ignore-aware nearest-neighbor metric.
        assert not (source_labels == 255).any() and not (target_labels == 255).any()
        object_counts = {
            str(int(label)): int((target_labels == label).sum())
            for label in torch.unique(target_labels)
            if label != 0
        }
        details.append(
            {
                "sequence": row["sequence"],
                "target_slot": row["target_slot"],
                "method": row["method"],
                "target_object_token_counts": object_counts,
                "nn_changed_fraction": float((old_nn != new_nn).float().mean()),
                "label_changed_fraction": float(
                    (source_labels[old_nn] != source_labels[new_nn]).float().mean()
                ),
                "mean_feature_cosine_drift": float(
                    (1 - (original_flat * adapted_flat).sum(-1)).mean()
                ),
                "relative_feature_change": float(
                    (adapted - target.features).norm() / target.features.norm()
                ),
            }
        )
    result = {
        "schema": "mcss.vision_probe.posthoc_diagnostic.v1",
        "scope": "Post hoc inspection of frozen validation outputs; no fitting or tuning",
        "per_pair": details,
        "methods": {},
    }
    for method in sorted(methods):
        group = [row for row in details if row["method"] == method]
        result["methods"][method] = {
            field: float(np.mean([row[field] for row in group]))
            for field in (
                "nn_changed_fraction",
                "label_changed_fraction",
                "mean_feature_cosine_drift",
                "relative_feature_change",
            )
        }
    (directory / "posthoc_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["methods"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.output, args.data_root)
