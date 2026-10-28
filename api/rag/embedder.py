"""Embedding backends. The real one is fastembed (ONNX) running BAAI/bge-small-en-v1.5
locally -- no API key, no network after the first model download."""
from __future__ import annotations
from typing import Protocol

import numpy as np

from . import config


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float32, rows L2-normalised so dot product == cosine similarity."""
        ...


class FastEmbedder:
    def __init__(self, model: str = config.EMBED_MODEL):
        self.name = model
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as e:  # explicit blocker, no silent fallback
                raise RuntimeError("fastembed is not installed: pip install fastembed") from e
            try:
                self._model = TextEmbedding(self.name, cache_dir=config.MODEL_CACHE)
            except Exception as e:
                raise RuntimeError(
                    f"could not load embedding model {self.name!r} (first run downloads it from "
                    f"Hugging Face into {config.MODEL_CACHE}): {e}"
                ) from e
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(list(self._load().embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)
