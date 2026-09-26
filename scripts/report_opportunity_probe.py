"""Generate the final opportunity report from saved raw-analysis artifacts."""

import argparse
from pathlib import Path

from mcss.vision_probe.opportunity_reporting import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_report(args.report_dir, args.work_dir)
    print(result["terminal_summary"], end="")


if __name__ == "__main__":
    main()
