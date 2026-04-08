"""
Embedding model wrapper using sentence-transformers.

Generates dense vector embeddings for text chunks and queries.
The model is loaded once and cached as a module-level singleton.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_model_instance: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        with _lock:
            if _model_instance is None:
                logger.info("Loading embedding model: %s", settings.embedding_model)
                _model_instance = SentenceTransformer(settings.embedding_model)
                logger.info("Embedding model loaded.")
    return _model_instance


class EmbeddingModel:
    """Wraps SentenceTransformer with helpers for RAG use-cases."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or settings.embedding_model
        # Pre-load so the first real call is fast
        self._model = _get_model()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Returns a Python list of floats."""
        vec: np.ndarray = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embed a list of strings efficiently.
        Returns a list of float lists (one per input text).
        """
        logger.debug("Embedding batch of %d texts", len(texts))
        vecs: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vecs.tolist()

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Cosine similarity between two pre-computed embeddings."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
