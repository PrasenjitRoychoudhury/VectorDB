import numpy as np
import pytest

from vectordb.index.brute_force import BruteForceIndex


def _unit(rng, n, dim):
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_add_and_search_returns_self_as_nearest():
    rng = np.random.default_rng(1)
    idx = BruteForceIndex(dim=8)
    vecs = _unit(rng, 20, 8)
    idx.add(list(range(20)), vecs)

    result = idx.search(vecs[5], k=1, ef=0)
    assert result[0][0] == 5
    assert result[0][1] < 1e-5


def test_duplicate_id_raises():
    idx = BruteForceIndex(dim=4)
    idx.add([1], np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        idx.add([1], np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))


def test_delete_excludes_from_search():
    rng = np.random.default_rng(2)
    idx = BruteForceIndex(dim=8)
    vecs = _unit(rng, 10, 8)
    idx.add(list(range(10)), vecs)
    idx.delete([3])

    results = idx.search(vecs[3], k=1, ef=0)
    assert results[0][0] != 3


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    idx = BruteForceIndex(dim=8)
    vecs = _unit(rng, 5, 8)
    idx.add(list(range(5)), vecs)
    idx.delete([2])

    path = tmp_path / "brute.npz"
    idx.save(path)

    idx2 = BruteForceIndex(dim=8)
    idx2.load(path)
    assert idx2.count == idx.count
    assert np.allclose(idx2._vectors, idx._vectors)
