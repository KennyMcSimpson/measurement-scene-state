#!/usr/bin/env python3
"""Run the score-independent static geometry audit for any compatible manifest."""

import argparse
import json

from mcss.mechanism_pilot.static_geometry import audit_manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metadata-csv")
    a = p.parse_args()
    print(
        json.dumps(
            audit_manifest(a.manifest, a.checkpoint, a.output_dir, metadata_csv=a.metadata_csv),
            indent=2,
        )
    )
