"""Audit source-to-input RGB without carrier inference or frame filtering."""

import argparse
import json

from mcss.mechanism_pilot.rgb_audit import audit_rgb_chain

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-metadata")
    args = parser.parse_args()
    print(json.dumps(audit_rgb_chain(args.manifest, args.output, args.official_metadata), indent=2))
