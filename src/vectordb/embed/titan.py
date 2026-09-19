"""Only module allowed to import boto3. Design doc, "Module: embed/".

Titan v2 has no native batch endpoint: fan out with asyncio.gather bounded
by a Semaphore. Every embedding is L2-normalised here — this is where the
index/'s "caller normalises" contract gets discharged for the texts path.
"""
import asyncio
import json
from typing import Any

import boto3
import numpy as np
from botocore.exceptions import ClientError

from vectordb.config import settings


class TitanEmbeddingError(Exception):
    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


class TitanEmbedder:
    def __init__(self, region_name: str | None = None):
        self._client = boto3.client("bedrock-runtime", region_name=region_name or settings.aws_region)
        self._semaphore = asyncio.Semaphore(settings.embed_concurrency)

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        vectors = await asyncio.gather(*(self._embed_one(t) for t in texts))
        return np.stack(vectors).astype(np.float32)

    async def _embed_one(self, text: str, max_attempts: int = 3, base_delay: float = 0.5) -> np.ndarray:
        async with self._semaphore:
            attempt = 0
            while True:
                try:
                    return await asyncio.to_thread(self._invoke_sync, text)
                except ClientError as e:
                    code = e.response.get("Error", {}).get("Code", "")
                    attempt += 1
                    if code == "ThrottlingException" and attempt < max_attempts:
                        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                        continue
                    raise TitanEmbeddingError(str(e), code=code) from e

    def _invoke_sync(self, text: str) -> np.ndarray:
        body = json.dumps({"inputText": text, "dimensions": settings.dim, "normalize": False})
        response = self._client.invoke_model(
            modelId=settings.embed_model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload: dict[str, Any] = json.loads(response["body"].read())
        vec = np.array(payload["embedding"], dtype=np.float32)
        if vec.shape[0] != settings.dim:
            raise TitanEmbeddingError(
                f"Titan returned dim {vec.shape[0]}, expected {settings.dim} — check config.dim / model version"
            )
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
