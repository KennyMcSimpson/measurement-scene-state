"""Nonblocking advisory process locks for dataset maintenance scripts.

Windows retains the original one-byte msvcrt lock. POSIX uses an exclusive
flock on the open file description. Closing the handle releases either lock;
lock files may remain on disk and can safely be reused after a process exits.
"""

from __future__ import annotations

import os
from typing import BinaryIO


def acquire_file_lock(handle: BinaryIO) -> None:
    """Lock an open, nonempty file without waiting; contention raises OSError."""
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def release_file_lock(handle: BinaryIO) -> None:
    """Release a previously acquired file lock without deleting its file."""
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
