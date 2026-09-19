"""Boot-time recovery sequence. Design doc, "Recovery sequence".
1. snapshot.read() if snapshot.vdb exists -> gives lsn
2. wal.replay(wal.log), skip records with offset < lsn, apply to Index
3. no snapshot -> lsn=0, replay the whole WAL
4. torn tail found -> log WARN, truncate_at(), continue serving
"""
from pathlib import Path

from vectordb.index.hnsw import HnswIndex


def recover(data_dir: Path, dim: int) -> HnswIndex:
    raise NotImplementedError("Day 10/11 — wired up once wal.py + snapshot.py + hnsw.py land")
