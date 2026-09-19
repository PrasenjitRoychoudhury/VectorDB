"""Confirms the api/ swap to HnswIndex actually works end-to-end through
the HTTP layer — the unit tests in test_hnsw*.py exercise HnswIndex
directly, this exercises it through /vectors, /search, /vectors/delete."""
import numpy as np
from fastapi.testclient import TestClient

from vectordb.api.main import app


def _unit_vec(seed: int, dim: int = 256) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    v /= np.linalg.norm(v)
    return v.tolist()


def test_add_then_search_finds_self():
    with TestClient(app) as client:
        v = _unit_vec(1)
        r = client.post("/vectors", json={"ids": [1], "vectors": [v]})
        assert r.status_code == 201
        assert r.json() == {"added": 1}

        r = client.post("/search", json={"vector": v, "k": 1})
        assert r.status_code == 200
        results = r.json()["results"]
        assert results[0]["id"] == 1
        assert results[0]["distance"] < 1e-4


def test_stats_reflects_real_hnsw_layers():
    with TestClient(app) as client:
        for i in range(50):
            client.post("/vectors", json={"ids": [i], "vectors": [_unit_vec(i)]})

        r = client.get("/stats")
        stats = r.json()
        assert stats["count"] == 50
        assert stats["dim"] == 256
        assert stats["layers"] >= 1  # real HNSW graph, not the brute-force stub's hardcoded 0


def test_delete_then_search_excludes_id():
    with TestClient(app) as client:
        v = _unit_vec(7)
        client.post("/vectors", json={"ids": [7], "vectors": [v]})
        client.post("/vectors", json={"ids": [8], "vectors": [_unit_vec(8)]})

        r = client.post("/vectors/delete", json={"ids": [7]})
        assert r.status_code == 200
        assert r.json() == {"deleted": 1}

        r = client.post("/search", json={"vector": v, "k": 2})
        ids = [res["id"] for res in r.json()["results"]]
        assert 7 not in ids


def test_non_unit_vector_rejected():
    with TestClient(app) as client:
        bad = [1.0] * 256  # norm way off 1.0
        r = client.post("/vectors", json={"ids": [1], "vectors": [bad]})
        assert r.status_code == 400


def test_duplicate_id_rejected():
    with TestClient(app) as client:
        v = _unit_vec(3)
        client.post("/vectors", json={"ids": [3], "vectors": [v]})
        r = client.post("/vectors", json={"ids": [3], "vectors": [v]})
        assert r.status_code == 409
