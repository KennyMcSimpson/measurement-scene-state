"""Execute the train-only small dynamic carrier engineering pilot."""

import argparse
import json
from pathlib import Path

from mcss.mechanism_pilot.small_training import SmallTrainingConfig, run_small_training
from mcss.mechanism_pilot.spatial import SPATIAL_MODES, carrier_config_for_spatial_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--spatial-mode", choices=SPATIAL_MODES, default="legacy-forward")
    args = parser.parse_args()
    config = SmallTrainingConfig(**json.loads(args.config.read_text())) if args.config else None
    print(
        json.dumps(
            run_small_training(
                args.manifest,
                args.output_dir,
                device=args.device,
                config=config,
                carrier_config=carrier_config_for_spatial_mode(args.spatial_mode),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
