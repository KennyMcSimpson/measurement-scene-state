"""Run the preregistered four frozen static development diagnostics; never holdout."""

import argparse

from mcss.mechanism_pilot.support_redesign_runner import run_development

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("experiment", "dev-manifest", "exposed-manifest", "checkpoint", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    a = p.parse_args()
    rows = run_development(
        a.experiment, a.dev_manifest, a.exposed_manifest, a.checkpoint, a.output, device=a.device
    )
    print(f"Completed {len(rows)} static rows; final holdout not accessed")
