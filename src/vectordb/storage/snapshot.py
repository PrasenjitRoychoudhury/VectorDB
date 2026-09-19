"""Snapshot format. Design doc, Section 4.
Header: magic b"VDB1" | version u16=1 | dim u16 | M u16 | count u32
        | entry i64 (-1 if empty) | lsn u64
Then: tombstone bitmap, level array, adjacency lists per level, vector block.

Atomic write: temp file + fsync + os.replace(). Version checked before
anything else is parsed — no upcast logic in v1 by design.
"""
from pathlib import Path

MAGIC = b"VDB1"
VERSION = 1


def write(path: Path, header: dict, tombstones, levels, layers, vectors) -> None:
    raise NotImplementedError("Day 11")


def read(path: Path) -> dict:
    raise NotImplementedError("Day 11 — validates MAGIC and VERSION before parsing further")
