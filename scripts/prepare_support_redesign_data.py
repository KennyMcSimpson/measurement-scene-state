#!/usr/bin/env python3
"""Prepare sealed development/final holdout cohorts without running model outputs."""

import argparse

from mcss.mechanism_pilot.support_redesign_data import prepare_support_redesign

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
    result = prepare_support_redesign(
        a.output,
        a.experiment,
        a.checkpoint,
        a.training_dir,
        a.training_manifest,
        a.partitions,
        a.calibration,
    )
    print(result["DATA_COHORT_STATUS"], len(result["scenes"]))
