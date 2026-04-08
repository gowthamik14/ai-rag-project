"""
FastAPI route definitions.

Singletons (_pipeline, _router_wf) are created lazily on first request
so the app starts cleanly even when Firestore or Ollama are not yet available.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from .schemas import (
    ChatRequest,
    HealthResponse,
    IngestFileRequest,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RouteRequest,
    RouteResponse,
    SourceChunk,
    SummarizeRequest,
    SummarizeResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Lazy singletons — created on first use, not at import time
# ------------------------------------------------------------------ #
_pipeline = None
_router_wf = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from rag.pipeline import RAGPipeline
        _pipeline = RAGPipeline()
    return _pipeline


def _get_router_wf():
    global _router_wf
    if _router_wf is None:
        from workflows import WorkflowRouter
        _router_wf = WorkflowRouter()
    return _router_wf


# ------------------------------------------------------------------ #
# Routers
# ------------------------------------------------------------------ #
health_router   = APIRouter(tags=["Health"])
ingest_router   = APIRouter(prefix="/ingest",    tags=["Ingestion"])
query_router    = APIRouter(prefix="/query",     tags=["Q&A"])
chat_router     = APIRouter(prefix="/chat",      tags=["Chat"])
summarize_router = APIRouter(prefix="/summarize", tags=["Summarization"])
route_router    = APIRouter(prefix="/route",     tags=["Router"])


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #

@health_router.get("/health", response_model=HealthResponse)
def health_check():
    try:
        info = _get_pipeline().health()
        return HealthResponse(
            status="ok",
            vector_store_total=info["vector_store_total"],
            llm_available=info["llm_available"],
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return HealthResponse(status="degraded", vector_store_total=0, llm_available=False)


# ------------------------------------------------------------------ #
# Ingestion
# ------------------------------------------------------------------ #

@ingest_router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_text(req: IngestTextRequest):
    from workflows import IngestionWorkflow
    result = IngestionWorkflow().run({
        "text":     req.text,
        "doc_id":   req.doc_id,
        "metadata": req.metadata,
    })
    if result.status.value == "failed":
        raise HTTPException(status_code=422, detail=result.errors)
    return IngestResponse(**result.output)


@ingest_router.post("/file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_file(req: IngestFileRequest):
    from workflows import IngestionWorkflow
    result = IngestionWorkflow().run({
        "file_path": req.file_path,
        "doc_id":    req.doc_id,
        "metadata":  req.metadata,
    })
    if result.status.value == "failed":
        raise HTTPException(status_code=422, detail=result.errors)
    return IngestResponse(**result.output)


# ------------------------------------------------------------------ #
# Q&A
# ------------------------------------------------------------------ #

@query_router.post("", response_model=QueryResponse)
def query(req: QueryRequest):
    result = _get_pipeline().query(
        question=req.question,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        blocked=result.blocked,
        block_reason=result.block_reason,
        sources=[
            SourceChunk(
                score=c.get("score", 0.0),
                source=c.get("source", c.get("doc_id", "")),
                chunk_index=c.get("chunk_index"),
                text_preview=c.get("text", "")[:200],
            )
            for c in result.retrieved_chunks
        ],
    )


@query_router.post("/stream")
def query_stream(req: QueryRequest):
    """Server-sent event stream for token-by-token answers."""
    def token_generator():
        for token in _get_pipeline().stream_query(
            question=req.question,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
        ):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


# ------------------------------------------------------------------ #
# Chat (conversational)
# ------------------------------------------------------------------ #

@chat_router.post("", response_model=QueryResponse)
def chat(req: ChatRequest):
    pipeline   = _get_pipeline()
    session_id = req.session_id

    if not session_id:
        session_id = pipeline.create_session(user_id=req.user_id)
        logger.info("Created new session: %s", session_id)

    result = pipeline.chat(
        question=req.question,
        session_id=session_id,
        top_k=req.top_k,
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        session_id=result.session_id,
        blocked=result.blocked,
        block_reason=result.block_reason,
        sources=[
            SourceChunk(
                score=c.get("score", 0.0),
                source=c.get("source", c.get("doc_id", "")),
                chunk_index=c.get("chunk_index"),
                text_preview=c.get("text", "")[:200],
            )
            for c in result.retrieved_chunks
        ],
    )


# ------------------------------------------------------------------ #
# Summarization
# ------------------------------------------------------------------ #

@summarize_router.post("", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    from workflows import SummarizationWorkflow
    inputs = {"style": req.style, "max_length": req.max_length}
    if req.doc_id:
        inputs["doc_id"] = req.doc_id
    if req.text:
        inputs["text"] = req.text

    result = SummarizationWorkflow().run(inputs)
    if result.status.value == "failed":
        raise HTTPException(status_code=422, detail=result.errors)
    return SummarizeResponse(**result.output)


# ------------------------------------------------------------------ #
# Intent-based router
# ------------------------------------------------------------------ #

@route_router.post("", response_model=RouteResponse)
def route(req: RouteRequest):
    inputs = dict(req.payload)
    if req.intent:
        inputs["intent"] = req.intent
    result = _get_router_wf().route(inputs)
    return RouteResponse(**result.to_dict())
