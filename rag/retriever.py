"""
Retriever: embeds a query and fetches the top-k most relevant chunks
from the FAISS vector store, then optionally enriches each result with
the full chunk text from Firestore.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore
from db.document_store import DocumentStore
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        document_store: Optional[DocumentStore] = None,
    ) -> None:
        self._embedder = embedding_model or EmbeddingModel()
        self._vector_store = vector_store or VectorStore()
        self._doc_store = document_store or DocumentStore()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: float = 0.0,
        enrich_from_firestore: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Embed `query`, search FAISS, and return ranked chunks.

        enrich_from_firestore: if True, re-fetches each chunk from Firestore
        to guarantee the latest text (useful when chunks are updated after indexing).
        """
        query_vec = self._embedder.embed_text(query)
        results = self._vector_store.search(
            query_embedding=query_vec,
            top_k=top_k or settings.top_k_retrieval,
            score_threshold=score_threshold,
        )

        if enrich_from_firestore:
            results = self._enrich(results)

        logger.info(
            "Retrieved %d chunks for query (len=%d, top_k=%s)",
            len(results),
            len(query),
            top_k,
        )
        return results

    def _enrich(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched = []
        for item in results:
            chunk_id = item.get("chunk_id")
            if chunk_id:
                fresh = self._doc_store.get_chunk(chunk_id)
                if fresh:
                    item["text"] = fresh.get("text", item.get("text", ""))
            enriched.append(item)
        return enriched

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a single context string from retrieved chunks to inject into prompts.
        Each chunk is numbered and separated by a divider.
        """
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "").strip()
            source = chunk.get("source", chunk.get("doc_id", "unknown"))
            parts.append(f"[{i}] (source: {source})\n{text}")
        return "\n\n---\n\n".join(parts)
