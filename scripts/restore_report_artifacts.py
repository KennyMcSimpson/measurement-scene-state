"""Restore losslessly compressed report text; never download datasets or overwrite changes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def restore(root=ROOT, *, check=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "docs/report_artifacts.json").read_text())
    for row in manifest["artifacts"]:
        target, archive = ((root / row[key]).resolve() for key in ("path", "archive"))
        for path in (target, archive):
            if not path.is_relative_to(root / "docs/experiments"):
                raise ValueError("Report path escapes experiment directory")
        compressed = archive.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != row["compressed_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        content = gzip.decompress(compressed)
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Report checksum mismatch: {target}")
        if target.exists():
            if target.read_bytes() != content:
                raise FileExistsError(f"Refusing to replace changed report: {target}")
        elif not check:
            with target.open("xb") as handle:
                handle.write(content)
        print(("Verified " if check else "Ready ") + str(target.relative_to(root)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify archives without extracting")
    restore(check=parser.parse_args().check)
