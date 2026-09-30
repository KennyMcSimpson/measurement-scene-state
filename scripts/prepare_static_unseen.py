#!/usr/bin/env python3
"""Prepare and seal metadata-selected unseen development data, without model outputs."""

import argparse

from mcss.mechanism_pilot.unseen_data import prepare_unseen

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in (
        "output",
        "experiment",
        "checkpoint",
        "training-dir",
        "training-manifest",
        "partitions",
        "calibration",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    result = prepare_unseen(
        a.output,
        a.experiment,
        a.checkpoint,
        a.training_dir,
        a.training_manifest,
        a.partitions,
        a.calibration,
    )
    print(result["UNSEEN_DEV_STATUS"], len(result["scenes"]))
