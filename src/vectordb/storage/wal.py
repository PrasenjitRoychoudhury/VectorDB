"""Write-ahead log. Design doc, Section 4 + "storage/ module".
Record: op:u8 | id:u64 | payload_len:u32 | payload | crc32:u32
op = 1 ADD (payload = dim x float32), op = 2 DELETE (payload = empty).
CRC32 over everything before the crc field itself.

fsync runs on a dedicated single-worker ThreadPoolExecutor so it never blocks
the FastAPI event loop (Amendment #3) — a 1-worker pool is already a serial
queue, so this doubles as write ordering, no extra lock needed here.
"""
import asyncio
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, NamedTuple

_HEADER = struct.Struct("<BQI")  # op, id, payload_len
_CRC = struct.Struct("<I")

OP_ADD = 1
OP_DELETE = 2


class WalRecord(NamedTuple):
    op: int
    id: int
    payload: bytes


class WriteAheadLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "ab", buffering=0)
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def append(self, op: int, id_: int, payload: bytes) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._write_sync, op, id_, payload)

    def _write_sync(self, op: int, id_: int, payload: bytes) -> None:
        body = _HEADER.pack(op, id_, len(payload)) + payload
        crc = zlib.crc32(body)
        record = body + _CRC.pack(crc)
        self._fh.write(record)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()
        self._executor.shutdown(wait=True)

    @staticmethod
    def replay_with_offset(path: Path) -> tuple[list[WalRecord], int]:
        """Parses sequentially; stops — does not raise — at the first bad
        CRC, per contract (a torn tail after a crash is expected). Returns
        (records, offset) where offset is how many leading bytes were
        successfully validated, so a caller can truncate the torn tail."""
        if not path.exists():
            return [], 0
        with open(path, "rb") as f:
            data = f.read()
        offset = 0
        records: list[WalRecord] = []
        while offset < len(data):
            if offset + _HEADER.size > len(data):
                break  # torn header
            op, id_, payload_len = _HEADER.unpack_from(data, offset)
            record_end = offset + _HEADER.size + payload_len + _CRC.size
            if record_end > len(data):
                break  # torn payload/crc
            payload = data[offset + _HEADER.size: offset + _HEADER.size + payload_len]
            (crc_stored,) = _CRC.unpack_from(data, offset + _HEADER.size + payload_len)
            body = data[offset:offset + _HEADER.size + payload_len]
            if zlib.crc32(body) != crc_stored:
                break  # torn/corrupt record
            records.append(WalRecord(op, id_, payload))
            offset = record_end
        return records, offset

    @staticmethod
    def replay(path: Path) -> Iterator[WalRecord]:
        records, _ = WriteAheadLog.replay_with_offset(path)
        yield from records

    @staticmethod
    def truncate_at(path: Path, offset: int) -> None:
        with open(path, "r+b") as f:
            f.truncate(offset)
