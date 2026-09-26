"""Regenerate V2 statistics from saved candidate/decision JSON, never media."""

import argparse
import json
from pathlib import Path

from mcss.vision_probe.v2_reporting import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--v1-report-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.report_dir, args.work_dir, args.v1_report_dir), indent=2))


if __name__ == "__main__":
    main()
