"""FastAPI app. Design doc, "Module: api/".

Day 2/6 state: backed by BruteForceIndex, no WAL/storage wired in yet
(that lands Day 10-11 via storage/recovery.py). Routes and the concurrency
model (lock on writes, lock-free reads) are the real, final shape already —
only the index implementation underneath gets swapped for HnswIndex later.
"""
import asyncio
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

from vectordb.config import settings
from vectordb.embed.titan import TitanEmbeddingError, TitanEmbedder
from vectordb.index.brute_force import BruteForceIndex

_NORM_TOL = 1e-3


class AppState:
    index: BruteForceIndex
    embedder: TitanEmbedder
    write_lock: asyncio.Lock


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.index = BruteForceIndex(dim=settings.dim)
    state.embedder = TitanEmbedder()
    state.write_lock = asyncio.Lock()
    yield


app = FastAPI(title="vectordb", lifespan=lifespan)


# ---- request/response models ----

class VectorsIn(BaseModel):
    ids: list[int]
    vectors: list[list[float]] | None = None
    texts: list[str] | None = None

    @model_validator(mode="after")
    def one_of_vectors_or_texts(self):
        if (self.vectors is None) == (self.texts is None):
            raise ValueError("exactly one of `vectors` or `texts` must be supplied")
        if self.vectors is not None and len(self.vectors) != len(self.ids):
            raise ValueError("len(vectors) must match len(ids)")
        if self.texts is not None and len(self.texts) != len(self.ids):
            raise ValueError("len(texts) must match len(ids)")
        return self


class SearchIn(BaseModel):
    vector: list[float] | None = None
    text: str | None = None
    k: int = 10
    ef: int = settings.ef_search_default

    @model_validator(mode="after")
    def one_of_vector_or_text(self):
        if (self.vector is None) == (self.text is None):
            raise ValueError("exactly one of `vector` or `text` must be supplied")
        return self


class DeleteIn(BaseModel):
    ids: list[int]


# ---- routes ----

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    idx = state.index
    return {
        "count": idx.count,
        "dim": idx.dim,
        "layers": 0,  # brute-force has no layers; meaningful once HnswIndex is wired in (Day 8+)
        "bytes": int(idx._vectors.nbytes),
        "tombstoned": len(idx._tombstoned),
    }


@app.post("/vectors", status_code=201)
async def add_vectors(body: VectorsIn):
    if body.vectors is not None:
        vecs = np.array(body.vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORM_TOL):
            raise HTTPException(status_code=400, detail="vectors must be L2-normalised (unit norm)")
    else:
        try:
            vecs = await state.embedder.embed_batch(body.texts)
        except TitanEmbeddingError as e:
            raise HTTPException(status_code=502, detail=f"embedding failed: {e.code or 'unknown'}: {e}")

    async with state.write_lock:
        try:
            state.index.add(body.ids, vecs)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    return {"added": len(body.ids)}


@app.post("/search")
async def search(body: SearchIn):
    if body.vector is not None:
        q = np.array(body.vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if abs(norm - 1.0) > _NORM_TOL:
            raise HTTPException(status_code=400, detail="vector must be L2-normalised (unit norm)")
    else:
        try:
            q = (await state.embedder.embed_batch([body.text]))[0]
        except TitanEmbeddingError as e:
            raise HTTPException(status_code=502, detail=f"embedding failed: {e.code or 'unknown'}: {e}")

    ef_eff = max(body.ef, body.k)  # Amendment #4 clamp
    results = state.index.search(q, k=body.k, ef=ef_eff)
    return {"results": [{"id": i, "distance": d} for i, d in results]}


@app.post("/vectors/delete")  # Amendment #2 — not DELETE /vectors, body-on-DELETE is fragile
async def delete_vectors(body: DeleteIn):
    async with state.write_lock:
        state.index.delete(body.ids)
    return {"deleted": len(body.ids)}
