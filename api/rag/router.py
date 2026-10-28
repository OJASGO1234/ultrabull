"""Read-only retrieval endpoints. Ingestion is CLI-only (python -m api.rag ingest)."""
from __future__ import annotations
import anthropic
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from . import config
from .embedder import FastEmbedder
from .generate import ask
from .retrieve import search
from .store import Store

router = APIRouter(prefix="/api/rag", tags=["rag"])
_embedder = FastEmbedder()  # model loads lazily on the first search, not at import


@router.get("/status")
def rag_status():
    store = Store()
    try:
        return {"model": store.get_meta("model"), "chunks": store.count(),
                "documents": [{"source": s, "chunks": n} for s, n in store.sources()]}
    finally:
        store.close()


@router.get("/search")
def rag_search(q: str = Query(..., min_length=1, max_length=500),
               k: int = Query(config.DEFAULT_TOP_K, ge=1, le=20),
               min_score: float = Query(config.DEFAULT_MIN_SCORE, ge=-1.0, le=1.0)):
    store = Store()
    try:
        if store.count() == 0:
            return {"query": q, "results": [], "note": "knowledge base is empty: run `python -m api.rag ingest`"}
        try:
            hits = search(store, _embedder, q, k, min_score)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {"query": q, "results": [h.to_dict() for h in hits]}
    finally:
        store.close()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    k: int = Field(config.DEFAULT_TOP_K, ge=1, le=10)


@router.post("/ask")
def rag_ask(req: AskRequest):
    """Grounded answer with numbered citations; Claude only sees retrieved passages."""
    store = Store()
    try:
        if store.count() == 0:
            raise HTTPException(status_code=503, detail="knowledge base is empty: run `python -m api.rag ingest`")
        try:
            return ask(store, _embedder, anthropic.Anthropic(), req.question, req.k)
        except RuntimeError as e:  # embedding model unavailable / index-model mismatch
            raise HTTPException(status_code=503, detail=str(e))
        except (anthropic.AuthenticationError, TypeError):
            raise HTTPException(status_code=503, detail="Ask needs an Anthropic API key: set ANTHROPIC_API_KEY "
                                                        "in the server environment and restart the API.")
        except anthropic.RateLimitError:
            raise HTTPException(status_code=429, detail="Rate limited by the Claude API — try again shortly.")
        except anthropic.APIStatusError as e:
            raise HTTPException(status_code=502, detail=f"Claude API error ({e.status_code}): {e.message}")
        except anthropic.APIConnectionError:
            raise HTTPException(status_code=503, detail="Could not reach the Claude API — check network.")
    finally:
        store.close()
