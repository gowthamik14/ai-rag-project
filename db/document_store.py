"""
High-level document store built on top of FirestoreClient.

Persists:
  - Raw documents  (collection: rag_documents)
  - Text chunks    (collection: rag_chunks)
  - Chat sessions  (collection: rag_sessions)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .firestore_client import FirestoreClient
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentStore:
    """CRUD interface for RAG documents, chunks, and sessions."""

    def __init__(self) -> None:
        self._fs = FirestoreClient()
        self._docs_col = settings.firestore_documents_collection
        self._chunks_col = settings.firestore_chunks_collection
        self._sessions_col = settings.firestore_sessions_collection

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #

    def save_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Persist a raw document. Returns the document ID."""
        doc_id = doc_id or str(uuid.uuid4())
        record = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": _now_iso(),
        }
        self._fs.set(self._docs_col, doc_id, record)
        logger.info("Saved document %s (%d chars)", doc_id, len(content))
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._fs.get(self._docs_col, doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        return self._fs.list_all(self._docs_col)

    def delete_document(self, doc_id: str) -> None:
        self._fs.delete(self._docs_col, doc_id)
        # Also purge associated chunks
        chunks = self._fs.query(self._chunks_col, filters=[("doc_id", "==", doc_id)])
        for chunk in chunks:
            self._fs.delete(self._chunks_col, chunk["id"])
        logger.info("Deleted document %s and %d chunks", doc_id, len(chunks))

    # ------------------------------------------------------------------ #
    # Chunks
    # ------------------------------------------------------------------ #

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Bulk-save pre-built chunk dicts.
        Each chunk must have: id, doc_id, text, chunk_index.
        Returns number of chunks saved.
        """
        for chunk in chunks:
            chunk.setdefault("created_at", _now_iso())
        written = self._fs.batch_set(self._chunks_col, chunks, id_field="id")
        logger.info("Saved %d chunks to Firestore", written)
        return written

    def get_chunks_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        return self._fs.query(
            self._chunks_col,
            filters=[("doc_id", "==", doc_id)],
            order_by="chunk_index",
        )

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self._fs.get(self._chunks_col, chunk_id)

    # ------------------------------------------------------------------ #
    # Sessions  (conversation history)
    # ------------------------------------------------------------------ #

    def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self._fs.set(
            self._sessions_col,
            session_id,
            {
                "id": session_id,
                "user_id": user_id,
                "messages": [],
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            },
        )
        return session_id

    def append_message(self, session_id: str, role: str, content: str) -> None:
        session = self._fs.get(self._sessions_col, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session["messages"].append(
            {"role": role, "content": content, "timestamp": _now_iso()}
        )
        session["updated_at"] = _now_iso()
        self._fs.set(self._sessions_col, session_id, session)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._fs.get(self._sessions_col, session_id)

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        return session["messages"] if session else []
