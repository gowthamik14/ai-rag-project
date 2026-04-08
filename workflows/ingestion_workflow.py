"""
IngestionWorkflow
=================
Takes raw text (or a file path) and ingests it into the RAG system:

  1. load_text      – read content from string or file
  2. split_chunks   – split into overlapping chunks
  3. embed_chunks   – generate embeddings for each chunk
  4. store_firestore – persist doc + chunks to Firestore
  5. index_vectors  – add embeddings to FAISS

Inputs (dict)
-------------
  text        : str  (raw document text)          ← one of these required
  file_path   : str  (path to .txt / .pdf file)
  doc_id      : str  (optional – auto-generated if omitted)
  metadata    : dict (optional extra metadata)
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_workflow import BaseWorkflow
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore
from db.document_store import DocumentStore
from utils.text_splitter import TextSplitter
from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionWorkflow(BaseWorkflow):
    name = "ingestion"

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        document_store: Optional[DocumentStore] = None,
    ) -> None:
        self._embedder = embedding_model or EmbeddingModel()
        self._vector_store = vector_store or VectorStore()
        self._doc_store = document_store or DocumentStore()
        self._splitter = TextSplitter()

    # ------------------------------------------------------------------ #
    # Contract
    # ------------------------------------------------------------------ #

    def _validate(self, inputs: Dict[str, Any]) -> None:
        if not inputs.get("text") and not inputs.get("file_path"):
            raise ValueError("Either 'text' or 'file_path' must be provided.")

    @property
    def _steps(self):
        return [
            ("load_text", self._step_load_text),
            ("split_chunks", self._step_split_chunks),
            ("embed_chunks", self._step_embed_chunks),
            ("store_firestore", self._step_store_firestore),
            ("index_vectors", self._step_index_vectors),
        ]

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _step_load_text(self, ctx: Dict[str, Any]) -> None:
        inputs = ctx["inputs"]
        text = inputs.get("text", "")

        if not text and (fp := inputs.get("file_path")):
            path = Path(fp)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {fp}")
            if path.suffix.lower() == ".pdf":
                text = self._read_pdf(path)
            else:
                text = path.read_text(encoding="utf-8")

        ctx["raw_text"] = text.strip()
        ctx["doc_id"] = inputs.get("doc_id") or str(uuid.uuid4())
        ctx["metadata"] = {
            "source": inputs.get("file_path", "inline"),
            **(inputs.get("metadata") or {}),
        }
        logger.info("Loaded %d chars for doc_id=%s", len(ctx["raw_text"]), ctx["doc_id"])

    def _step_split_chunks(self, ctx: Dict[str, Any]) -> None:
        chunks = self._splitter.split(ctx["raw_text"])
        ctx["chunk_texts"] = chunks
        logger.info("Split into %d chunks", len(chunks))

    def _step_embed_chunks(self, ctx: Dict[str, Any]) -> None:
        texts = ctx["chunk_texts"]
        embeddings = self._embedder.embed_batch(texts)
        ctx["embeddings"] = embeddings
        logger.info("Generated %d embeddings", len(embeddings))

    def _step_store_firestore(self, ctx: Dict[str, Any]) -> None:
        doc_id = ctx["doc_id"]
        # Save raw document
        self._doc_store.save_document(
            content=ctx["raw_text"],
            metadata=ctx["metadata"],
            doc_id=doc_id,
        )
        # Save chunks
        chunk_records = [
            {
                "id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": text,
                "chunk_index": i,
                "source": ctx["metadata"].get("source", ""),
            }
            for i, text in enumerate(ctx["chunk_texts"])
        ]
        ctx["chunk_records"] = chunk_records
        self._doc_store.save_chunks(chunk_records)
        logger.info("Stored doc + %d chunks in Firestore", len(chunk_records))

    def _step_index_vectors(self, ctx: Dict[str, Any]) -> None:
        chunk_records = ctx["chunk_records"]
        embeddings = ctx["embeddings"]
        metadatas = [
            {
                "chunk_id": rec["id"],
                "doc_id": rec["doc_id"],
                "text": rec["text"],
                "source": rec.get("source", ""),
                "chunk_index": rec["chunk_index"],
            }
            for rec in chunk_records
        ]
        self._vector_store.add(embeddings, metadatas)

        ctx["output"] = {
            "doc_id": ctx["doc_id"],
            "chunks_indexed": len(embeddings),
            "total_vectors": self._vector_store.total_vectors,
        }
        logger.info("Indexed %d vectors (total: %d)", len(embeddings), self._vector_store.total_vectors)

    # ------------------------------------------------------------------ #
    # PDF helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ImportError("Install 'pypdf' to ingest PDF files: pip install pypdf")
