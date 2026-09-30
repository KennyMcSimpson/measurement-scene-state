"""Run static qualification; no history branches, dynamic actions, or controller."""

import argparse

from mcss.mechanism_pilot.static_evaluation import run_static

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--residual-checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--unseen", action="store_true")
    args = parser.parse_args()
    run_static(
        args.manifest,
        args.checkpoint,
        args.output,
        residual_path=args.residual_checkpoint,
        device=args.device,
        unseen=args.unseen,
    )
