"""
High-level document store built on top of FirestoreClient.

When Firestore is unavailable the store operates in local-only mode:
  - Documents and chunks are accepted but not persisted remotely.
  - Sessions are stored in-memory (lost on restart — fine for dev/testing).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .firestore_client import FirestoreClient
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# In-memory fallback store used when Firestore is unavailable
_mem_sessions: Dict[str, Dict[str, Any]] = {}
_mem_docs: Dict[str, Dict[str, Any]] = {}
_mem_chunks: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentStore:
    """CRUD interface for RAG documents, chunks, and sessions.
    Falls back to in-memory storage when Firestore is unavailable.
    """

    def __init__(self) -> None:
        self._fs         = FirestoreClient()
        self._docs_col   = settings.firestore_documents_collection
        self._chunks_col = settings.firestore_chunks_collection
        self._sessions_col = settings.firestore_sessions_collection

        if not self._fs.available:
            logger.warning(
                "DocumentStore running in local-only mode (no Firestore). "
                "Data will not be persisted between restarts."
            )

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #

    def save_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        doc_id = doc_id or str(uuid.uuid4())
        record = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": _now_iso(),
        }
        if self._fs.available:
            self._fs.set(self._docs_col, doc_id, record)
        else:
            _mem_docs[doc_id] = record
        logger.info("Saved document %s (%d chars)", doc_id, len(content))
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if self._fs.available:
            return self._fs.get(self._docs_col, doc_id)
        return _mem_docs.get(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        if self._fs.available:
            return self._fs.list_all(self._docs_col)
        return list(_mem_docs.values())

    def delete_document(self, doc_id: str) -> None:
        if self._fs.available:
            self._fs.delete(self._docs_col, doc_id)
            chunks = self._fs.query(self._chunks_col, filters=[("doc_id", "==", doc_id)])
            for chunk in chunks:
                self._fs.delete(self._chunks_col, chunk["id"])
        else:
            _mem_docs.pop(doc_id, None)
            to_delete = [k for k, v in _mem_chunks.items() if v.get("doc_id") == doc_id]
            for k in to_delete:
                _mem_chunks.pop(k, None)
        logger.info("Deleted document %s", doc_id)

    # ------------------------------------------------------------------ #
    # Chunks
    # ------------------------------------------------------------------ #

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        for chunk in chunks:
            chunk.setdefault("created_at", _now_iso())
        if self._fs.available:
            written = self._fs.batch_set(self._chunks_col, chunks, id_field="id")
        else:
            for chunk in chunks:
                _mem_chunks[chunk["id"]] = chunk
            written = len(chunks)
        logger.info("Saved %d chunks", written)
        return written

    def get_chunks_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        if self._fs.available:
            return self._fs.query(
                self._chunks_col,
                filters=[("doc_id", "==", doc_id)],
                order_by="chunk_index",
            )
        return sorted(
            [c for c in _mem_chunks.values() if c.get("doc_id") == doc_id],
            key=lambda c: c.get("chunk_index", 0),
        )

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        if self._fs.available:
            return self._fs.get(self._chunks_col, chunk_id)
        return _mem_chunks.get(chunk_id)

    # ------------------------------------------------------------------ #
    # Sessions  (conversation history)
    # ------------------------------------------------------------------ #

    def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        record = {
            "id": session_id,
            "user_id": user_id,
            "messages": [],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        if self._fs.available:
            self._fs.set(self._sessions_col, session_id, record)
        else:
            _mem_sessions[session_id] = record
        return session_id

    def append_message(self, session_id: str, role: str, content: str) -> None:
        if self._fs.available:
            session = self._fs.get(self._sessions_col, session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")
            session["messages"].append(
                {"role": role, "content": content, "timestamp": _now_iso()}
            )
            session["updated_at"] = _now_iso()
            self._fs.set(self._sessions_col, session_id, session)
        else:
            session = _mem_sessions.get(session_id)
            if session is None:
                # Auto-create missing session in local mode
                _mem_sessions[session_id] = {
                    "id": session_id, "messages": [],
                    "created_at": _now_iso(), "updated_at": _now_iso(),
                }
                session = _mem_sessions[session_id]
            session["messages"].append(
                {"role": role, "content": content, "timestamp": _now_iso()}
            )
            session["updated_at"] = _now_iso()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self._fs.available:
            return self._fs.get(self._sessions_col, session_id)
        return _mem_sessions.get(session_id)

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        return session["messages"] if session else []
