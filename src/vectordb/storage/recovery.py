"""Boot-time recovery sequence. Design doc, "Recovery sequence".
Day 10 scope: WAL only, no snapshot yet (that's Day 11 — snapshot.read()
would give a starting lsn to skip already-persisted records; without one,
lsn=0 and the whole WAL replays, exactly per the frozen contract's
"No snapshot means replay the whole WAL from zero.").
"""
import logging
from pathlib import Path

import numpy as np

from vectordb.config import settings
from vectordb.index.hnsw import HnswIndex
from vectordb.storage.wal import OP_ADD, OP_DELETE, WriteAheadLog

logger = logging.getLogger(__name__)


def recover(data_dir: Path, dim: int) -> HnswIndex:
    idx = HnswIndex(
        dim=dim,
        m=settings.m,
        m_l0=settings.m_l0,
        ef_construction=settings.ef_construction,
        rng_seed=settings.rng_seed,
    )

    wal_path = data_dir / "wal.log"
    if not wal_path.exists():
        return idx

    file_size = wal_path.stat().st_size
    records, good_offset = WriteAheadLog.replay_with_offset(wal_path)

    for record in records:
        if record.op == OP_ADD:
            vector = np.frombuffer(record.payload, dtype="<f4").reshape(1, -1)
            if record.id in idx.id_to_slot:
                # Single-writer WAL should never re-ADD a live id — defensive
                # skip rather than crashing recovery on an unexpected record.
                logger.warning("WAL replay: id %d already present, skipping duplicate ADD", record.id)
                continue
            idx.add([record.id], vector)
        elif record.op == OP_DELETE:
            idx.delete([record.id])
        else:
            logger.warning("WAL replay: unknown op %d for id %d, skipping", record.op, record.id)

    if good_offset < file_size:
        logger.warning(
            "WAL torn tail: %d of %d bytes replayed cleanly — truncating the rest (expected after a crash)",
            good_offset, file_size,
        )
        WriteAheadLog.truncate_at(wal_path, good_offset)

    return idx
