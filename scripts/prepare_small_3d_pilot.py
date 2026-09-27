"""Prepare the three frozen TRAIN scenes only; no evaluation or full ZIP downloads."""

import argparse

from mcss.mechanism_pilot.data_prep import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--partitions", default="configs/hypersim_er_partitions.csv")
    args = parser.parse_args()
    prepare(args.output, args.calibration, args.partitions)
