"""Single source of env-var truth. No path literals or magic numbers
anywhere else in the codebase — everything imports `settings` from here.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTORDB_")

    data_dir: Path = Path("/var/lib/vectordb")
    dim: int = 256
    aws_region: str = "us-east-1"

    # HNSW parameters (frozen contract, Section 3)
    m: int = 16
    m_l0: int = 32
    ef_construction: int = 200
    ef_search_default: int = 128  # 64 measured 0.842 recall@10 (fails 0.95 gate) — see test_hnsw_search_delete.py
    rng_seed: int = 42

    # embed/
    embed_model_id: str = "amazon.titan-embed-text-v2:0"
    embed_concurrency: int = 5

    # api/
    host: str = "0.0.0.0"
    port: int = 8080


settings = Settings()
