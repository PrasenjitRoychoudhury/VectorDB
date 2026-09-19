"""Every test that spins up the FastAPI app now triggers real WAL I/O at
startup (recovery) and on every write (Day 10). Without this fixture, that
would mean every test tries to create/write /var/lib/vectordb — wrong on a
dev machine, and a permissions error outside the container. Redirect it to
an isolated tmp_path automatically, for every test, no opt-in required.
"""
import pytest

from vectordb.config import settings


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "vectordb_data")
    yield
