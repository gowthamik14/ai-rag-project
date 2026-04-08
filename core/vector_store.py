"""
FAISS-backed vector store with JSON sidecar for chunk metadata.

Layout on disk:
  data/faiss_index/index.faiss   – FAISS flat-L2 index
  data/faiss_index/metadata.json – list of chunk metadata dicts (parallel to index rows)

The in-memory index and metadata list are kept in sync: row i in the FAISS
index always corresponds to metadata[i].
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Persistent FAISS vector store for RAG retrieval."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        self._dim = dimension or settings.embedding_dimension
        self._index_dir = Path(index_path or settings.faiss_index_path)
        self._index_file = self._index_dir / settings.faiss_index_file
        self._meta_file = self._index_dir / settings.faiss_metadata_file
        self._lock = threading.Lock()

        self._index: faiss.IndexFlatIP  # inner-product ≈ cosine when vecs are L2-normalised
        self._metadata: List[Dict[str, Any]] = []

        self._load_or_create()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """Add vectors with their metadata. Both lists must be the same length."""
        if len(embeddings) != len(metadatas):
            raise ValueError("embeddings and metadatas must have the same length")

        vectors = np.array(embeddings, dtype=np.float32)
        with self._lock:
            self._index.add(vectors)
            self._metadata.extend(metadatas)
            self._persist()

        logger.info("Added %d vectors to FAISS index (total: %d)", len(embeddings), self._index.ntotal)

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar chunks.

        Returns list of dicts: {score, chunk_id, doc_id, text, ...metadata}
        Scores are cosine similarities (0–1 range for normalised vectors).
        """
        k = top_k or settings.top_k_retrieval
        if self._index.ntotal == 0:
            logger.warning("Vector store is empty — no results returned.")
            return []

        k = min(k, self._index.ntotal)
        query = np.array([query_embedding], dtype=np.float32)

        with self._lock:
            scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < score_threshold:
                continue
            results.append({"score": float(score), **self._metadata[idx]})

        logger.debug("Vector search returned %d results (top_k=%d)", len(results), k)
        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all vectors belonging to a document.
        FAISS FlatIP does not support in-place deletion, so we rebuild the index.
        Returns number of vectors removed.
        """
        with self._lock:
            keep_indices = [
                i for i, m in enumerate(self._metadata) if m.get("doc_id") != doc_id
            ]
            removed = len(self._metadata) - len(keep_indices)

            if removed == 0:
                return 0

            kept_meta = [self._metadata[i] for i in keep_indices]
            kept_vecs = np.array(
                [self._index.reconstruct(i) for i in keep_indices], dtype=np.float32
            )

            self._index = faiss.IndexFlatIP(self._dim)
            if len(kept_vecs):
                self._index.add(kept_vecs)
            self._metadata = kept_meta
            self._persist()

        logger.info("Removed %d vectors for doc_id=%s", removed, doc_id)
        return removed

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load_or_create(self) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)

        if self._index_file.exists() and self._meta_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            with open(self._meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(
                "Loaded FAISS index (%d vectors) from %s",
                self._index.ntotal,
                self._index_dir,
            )
        else:
            self._index = faiss.IndexFlatIP(self._dim)
            self._metadata = []
            logger.info("Created new FAISS index (dim=%d) at %s", self._dim, self._index_dir)

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self._index_file))
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
