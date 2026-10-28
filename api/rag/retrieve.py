"""Semantic retrieval: cosine similarity of the query against every stored chunk."""
from __future__ import annotations
from dataclasses import asdict, dataclass

import numpy as np

from . import config
from .embedder import Embedder
from .store import Store


@dataclass
class Hit:
    score: float
    source: str
    heading: str
    page: int | None
    chunk_index: int
    text: str
    chunk_id: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d


def search(store: Store, embedder: Embedder, query: str, top_k: int = config.DEFAULT_TOP_K,
           min_score: float = config.DEFAULT_MIN_SCORE) -> list[Hit]:
    """Best `top_k` chunks scoring >= min_score, best first. [] for an empty index,
    a blank query, or a query nothing in the corpus is close to."""
    query = (query or "").strip()
    if not query or top_k <= 0:
        return []
    stored_model = store.get_meta("model")
    if stored_model and stored_model != embedder.name:
        raise RuntimeError(f"index was built with {stored_model!r}, query embedder is {embedder.name!r}")
    rows, mat = store.all_chunks()
    if not rows:
        return []
    scores = mat @ embedder.embed([query])[0]
    order = np.argsort(-scores)[:top_k]
    return [
        Hit(float(scores[i]), rows[i][1], rows[i][2], rows[i][3], rows[i][4], rows[i][5], rows[i][0])
        for i in order if scores[i] >= min_score
    ]
