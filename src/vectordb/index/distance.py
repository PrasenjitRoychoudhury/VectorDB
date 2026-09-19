"""Distance functions. Inputs are assumed unit-normalised by the caller
(the index never normalises — see the design doc's normalisation contract).
"""
import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: shape (dim,), unit-norm. Returns 1 - dot(a, b)."""
    return 1.0 - float(np.dot(a, b))


def cosine_distance_batch(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """q: shape (dim,), unit-norm. matrix: shape (N, dim), rows unit-norm.
    Returns shape (N,) distances, ascending order NOT guaranteed — caller sorts.
    """
    return 1.0 - matrix @ q
