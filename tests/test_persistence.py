import numpy as np
from fastapi.testclient import TestClient

from vectordb.api.main import app
from vectordb.storage.wal import WriteAheadLog


def _unit_vec(seed: int, dim: int = 256) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    v /= np.linalg.norm(v)
    return v.tolist()


def test_data_survives_a_restart():
    """The actual Day 10 point: what every EC2 container restart today lost.
    Two separate TestClient sessions against the same app == two separate
    process lifetimes, sharing only what's on disk between them."""
    v = _unit_vec(1)

    with TestClient(app) as client:
        r = client.post("/vectors", json={"ids": [1], "vectors": [v]})
        assert r.status_code == 201

    # "restart": re-enter lifespan — this is exactly what happens on
    # docker stop && docker rm && docker run
    with TestClient(app) as client:
        r = client.get("/stats")
        assert r.json()["count"] == 1

        r = client.post("/search", json={"vector": v, "k": 1})
        assert r.json()["results"][0]["id"] == 1


def test_delete_survives_a_restart():
    v1, v2 = _unit_vec(10), _unit_vec(11)

    with TestClient(app) as client:
        client.post("/vectors", json={"ids": [10, 11], "vectors": [v1, v2]})
        client.post("/vectors/delete", json={"ids": [10]})

    with TestClient(app) as client:
        r = client.get("/stats")
        assert r.json()["count"] == 1

        r = client.post("/search", json={"vector": v1, "k": 2})
        ids = [res["id"] for res in r.json()["results"]]
        assert 10 not in ids
        assert 11 in ids


def test_recovery_survives_a_torn_tail(tmp_path, monkeypatch):
    """Simulates a crash mid-write: a WAL with one good record followed by
    garbage bytes. Recovery must apply the good record and discard the rest,
    per the frozen contract — not raise, not lose the good data too."""
    from vectordb.config import settings
    from vectordb.storage.recovery import recover

    monkeypatch.setattr(settings, "data_dir", tmp_path / "torn_test")
    settings.data_dir.mkdir(parents=True)
    wal_path = settings.data_dir / "wal.log"

    v = np.array(_unit_vec(5), dtype=np.float32)
    good_record = (
        (1).to_bytes(1, "little")
        + (99).to_bytes(8, "little")
        + len(v.tobytes()).to_bytes(4, "little")
        + v.tobytes()
    )
    import zlib
    good_record += zlib.crc32(good_record).to_bytes(4, "little")

    garbage_tail = b"\xde\xad\xbe\xef" * 10  # not a valid record, no valid CRC

    wal_path.write_bytes(good_record + garbage_tail)
    original_size = wal_path.stat().st_size

    idx = recover(settings.data_dir, dim=256)

    assert idx.count == 1
    assert 99 in idx.id_to_slot
    # torn tail must have been truncated off the file
    assert wal_path.stat().st_size < original_size
    assert wal_path.stat().st_size == len(good_record)
