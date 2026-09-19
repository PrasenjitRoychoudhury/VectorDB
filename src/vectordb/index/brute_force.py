"""Brute-force exact search. Used as (a) the Day 2/6 stub API backend before
HNSW exists, and (b) the ground-truth baseline for the Day 9 recall@10 gate.
No I/O — save/load are simple numpy dumps, not the production snapshot format
(that lives in storage/snapshot.py and is HNSW-specific).
"""
from pathlib import Path

import numpy as np

from vectordb.index.distance import cosine_distance_batch


class BruteForceIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[int] = []
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._tombstoned: set[int] = set()

    @property
    def count(self) -> int:
        return len(self._ids) - len(self._tombstoned)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        existing = set(self._ids)
        for i in ids:
            if i in existing:
                raise ValueError(f"id {i} already exists")
        self._ids.extend(ids)
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, k: int, ef: int = 0) -> list[tuple[int, float]]:
        # ef is accepted for interface parity with HNSW; unused here — brute
        # force always considers every live vector.
        if not self._ids:
            return []
        distances = cosine_distance_batch(query, self._vectors)
        order = np.argsort(distances)
        results: list[tuple[int, float]] = []
        for idx in order:
            vid = self._ids[idx]
            if vid in self._tombstoned:
                continue
            results.append((vid, float(distances[idx])))
            if len(results) == k:
                break
        return results

    def delete(self, ids: list[int]) -> None:
        self._tombstoned.update(ids)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            ids=np.array(self._ids, dtype=np.int64),
            vectors=self._vectors,
            tombstoned=np.array(list(self._tombstoned), dtype=np.int64),
        )

    def load(self, path: Path) -> None:
        data = np.load(path)
        self._ids = data["ids"].tolist()
        self._vectors = data["vectors"]
        self._tombstoned = set(data["tombstoned"].tolist())
