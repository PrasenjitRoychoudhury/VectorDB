import numpy as np

from vectordb.index.distance import cosine_distance, cosine_distance_batch


def test_identical_vectors_zero_distance():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(cosine_distance(v, v)) < 1e-6


def test_orthogonal_vectors_unit_distance():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_distance(a, b) - 1.0) < 1e-6


def test_opposite_vectors_max_distance():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert abs(cosine_distance(a, b) - 2.0) < 1e-6


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    q = rng.normal(size=8).astype(np.float32)
    q /= np.linalg.norm(q)
    M = rng.normal(size=(5, 8)).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)

    batch = cosine_distance_batch(q, M)
    single = np.array([cosine_distance(q, M[i]) for i in range(5)])
    assert np.allclose(batch, single, atol=1e-6)
