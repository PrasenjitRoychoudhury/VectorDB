"""The frozen index/ interface (design doc, Section 2). Both BruteForceIndex
and the future HnswIndex implement this. No I/O, no HTTP — numpy + stdlib
only, anywhere under index/.
"""
from pathlib import Path
from typing import Protocol

import numpy as np


class Index(Protocol):
    dim: int
    count: int  # live vectors, excludes tombstoned

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        """vectors shape (len(ids), dim), float32, L2-normalised by caller.
        Re-adding an existing id raises ValueError."""
        ...

    def search(self, query: np.ndarray, k: int, ef: int) -> list[tuple[int, float]]:
        """query shape (dim,), float32, L2-normalised.
        Returns up to k (id, distance) ascending. Tombstoned ids excluded.
        Fewer than k results is valid."""
        ...

    def delete(self, ids: list[int]) -> None:
        """Tombstone. Unknown ids ignored silently."""
        ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
