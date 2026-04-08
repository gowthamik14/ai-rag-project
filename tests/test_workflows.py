"""
Tests for workflow classes using mocked dependencies.
"""
from unittest.mock import MagicMock, patch

import pytest

from workflows.base_workflow import WorkflowStatus
from workflows.ingestion_workflow import IngestionWorkflow
from workflows.qa_workflow import QAWorkflow
from workflows.summarization_workflow import SummarizationWorkflow
from core.llm import OllamaLLM
from rag.pipeline import RAGPipeline, RAGResult


# ------------------------------------------------------------------ #
# IngestionWorkflow
# ------------------------------------------------------------------ #

def test_ingestion_requires_text_or_file():
    wf = IngestionWorkflow(
        embedding_model=MagicMock(),
        vector_store=MagicMock(),
        document_store=MagicMock(),
    )
    result = wf.run({})
    assert result.status == WorkflowStatus.FAILED
    assert any("text" in e or "file_path" in e for e in result.errors)


def test_ingestion_success():
    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1, 0.2]] * 3
    vector_store = MagicMock()
    vector_store.total_vectors = 3
    doc_store = MagicMock()

    with patch(
        "workflows.ingestion_workflow.TextSplitter.split",
        return_value=["chunk1", "chunk2", "chunk3"],
    ):
        wf = IngestionWorkflow(
            embedding_model=embedder,
            vector_store=vector_store,
            document_store=doc_store,
        )
        result = wf.run({"text": "This is a test document with some content."})

    assert result.status == WorkflowStatus.SUCCESS
    assert result.output["chunks_indexed"] == 3


# ------------------------------------------------------------------ #
# QAWorkflow
# ------------------------------------------------------------------ #

def test_qa_requires_question():
    pipeline = MagicMock(spec=RAGPipeline)
    wf = QAWorkflow(pipeline=pipeline)
    result = wf.run({"question": ""})
    assert result.status == WorkflowStatus.FAILED


def test_qa_one_shot():
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.query.return_value = RAGResult(
        question="Q", answer="A", retrieved_chunks=[]
    )
    wf = QAWorkflow(pipeline=pipeline)
    result = wf.run({"question": "What is AI?"})
    assert result.status == WorkflowStatus.SUCCESS
    assert result.output["answer"] == "A"


# ------------------------------------------------------------------ #
# SummarizationWorkflow
# ------------------------------------------------------------------ #

def test_summarization_requires_input():
    wf = SummarizationWorkflow(llm=MagicMock(), document_store=MagicMock())
    result = wf.run({})
    assert result.status == WorkflowStatus.FAILED


def test_summarization_inline_text():
    llm = MagicMock(spec=OllamaLLM)
    llm.generate.return_value = "Short summary."
    wf = SummarizationWorkflow(llm=llm, document_store=MagicMock())
    result = wf.run({"text": "Long document text here. " * 10, "style": "concise"})
    assert result.status == WorkflowStatus.SUCCESS
    assert "summary" in result.output
