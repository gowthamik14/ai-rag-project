"""
Integration-style tests for the RAG pipeline using mocks.
No live Firestore / Ollama connection is needed.
"""
from unittest.mock import MagicMock, patch

import pytest

from rag.pipeline import RAGPipeline, RAGResult
from rag.retriever import Retriever
from rag.generator import Generator
from db.document_store import DocumentStore


def _make_pipeline():
    retriever = MagicMock(spec=Retriever)
    generator = MagicMock(spec=Generator)
    doc_store = MagicMock(spec=DocumentStore)

    retriever.retrieve.return_value = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "Paris is the capital of France.", "score": 0.95}
    ]
    retriever.format_context.return_value = "[1] Paris is the capital of France."
    generator.generate.return_value = "Paris is the capital of France."
    generator.chat_generate.return_value = "Paris is the capital of France."
    doc_store.create_session.return_value = "session-123"
    doc_store.get_session_history.return_value = []

    return RAGPipeline(retriever=retriever, generator=generator, document_store=doc_store)


def test_query_returns_result():
    pipeline = _make_pipeline()
    result = pipeline.query("What is the capital of France?")
    assert isinstance(result, RAGResult)
    assert "Paris" in result.answer
    assert len(result.retrieved_chunks) == 1


def test_query_passes_top_k():
    pipeline = _make_pipeline()
    pipeline.query("test question", top_k=3)
    pipeline._retriever.retrieve.assert_called_once()
    call_kwargs = pipeline._retriever.retrieve.call_args.kwargs
    assert call_kwargs.get("top_k") == 3


def test_create_session():
    pipeline = _make_pipeline()
    sid = pipeline.create_session(user_id="user-1")
    assert sid == "session-123"
    pipeline._doc_store.create_session.assert_called_once_with(user_id="user-1")


def test_chat_persists_messages():
    pipeline = _make_pipeline()
    pipeline.chat("Hello?", session_id="session-123")
    assert pipeline._doc_store.append_message.call_count == 2  # user + assistant


def test_stream_query_yields_tokens():
    pipeline = _make_pipeline()
    pipeline._generator.stream_generate = MagicMock(return_value=iter(["Par", "is"]))
    tokens = list(pipeline.stream_query("Capital of France?"))
    assert tokens == ["Par", "is"]
