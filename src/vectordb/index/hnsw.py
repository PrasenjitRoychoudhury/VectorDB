"""HNSW index. Design doc, Section 2 (index/ module) — implemented Day 8
(insert) and Day 9 (search). Not wired into api/ until then; BruteForceIndex
serves in the meantime.
"""
from pathlib import Path

import numpy as np

from vectordb.config import settings


class HnswIndex:
    def __init__(self, dim: int, m: int = settings.m, m_l0: int = settings.m_l0,
                 ef_construction: int = settings.ef_construction,
                 rng_seed: int = settings.rng_seed):
        self.dim = dim
        self.m = m
        self.m_l0 = m_l0
        self.ef_construction = ef_construction
        self.ml = 1.0 / np.log(m)
        self.rng = np.random.default_rng(rng_seed)

        # Data structures — see design doc, "hnsw.py — data structures"
        self._capacity = 0
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.id_to_slot: dict[int, int] = {}
        self.slot_to_id: list[int] = []
        self.free_slots: list[int] = []
        self.tombstoned: np.ndarray = np.empty(0, dtype=bool)
        self.levels: list[int] = []
        self.layers: list[dict[int, list[int]]] = []
        self.entry_point: int = -1

    @property
    def count(self) -> int:
        return len(self.id_to_slot)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        raise NotImplementedError("Day 8 — HNSW insert")

    def search(self, query: np.ndarray, k: int, ef: int) -> list[tuple[int, float]]:
        raise NotImplementedError("Day 9 — HNSW search")

    def delete(self, ids: list[int]) -> None:
        raise NotImplementedError("Day 8 — tombstone + entry_point reassignment")

    def to_arrays(self) -> dict:
        """Plain numpy/dict structures for storage/snapshot.py to persist.
        index/ never touches a file directly — see design doc module rule."""
        raise NotImplementedError("Day 11")

    @classmethod
    def from_arrays(cls, data: dict) -> "HnswIndex":
        raise NotImplementedError("Day 11")

    def save(self, path: Path) -> None:
        raise NotImplementedError("delegates to storage/snapshot.py — Day 11")

    def load(self, path: Path) -> None:
        raise NotImplementedError("delegates to storage/snapshot.py — Day 11")
