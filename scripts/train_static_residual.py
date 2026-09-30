"""Train the fixed-budget static residual control on checkpoint training scenes only."""

import argparse
import json

from mcss.mechanism_pilot.static_residual import ResidualTrainingConfig, train_residual

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--carrier-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    print(
        json.dumps(
            train_residual(
                args.manifest,
                args.carrier_checkpoint,
                args.output_dir,
                device=args.device,
                config=ResidualTrainingConfig(steps=args.steps),
            ),
            indent=2,
        )
    )
