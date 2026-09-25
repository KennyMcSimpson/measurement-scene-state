"""CLI entry point for the exploratory vision probe."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcss.vision_probe.experiment import (  # noqa: E402
    SequenceInfo,
    discover_sequences,
    mask_metrics,
    run,
    split_sequences,
)

__all__ = ["SequenceInfo", "discover_sequences", "mask_metrics", "run", "split_sequences", "main"]


def main() -> int:
    from mcss.vision_probe.experiment import main as experiment_main

    return experiment_main()


if __name__ == "__main__":
    raise SystemExit(main())
