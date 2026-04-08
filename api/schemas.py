"""
Pydantic request / response schemas for the REST API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Ingest
# ------------------------------------------------------------------ #

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to ingest.")
    doc_id: Optional[str] = Field(None, description="Optional document ID (auto-generated if omitted).")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="Server-side path to a .txt or .pdf file.")
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    total_vectors: int


# ------------------------------------------------------------------ #
# Q&A
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)


class SourceChunk(BaseModel):
    score: float
    source: str
    chunk_index: Optional[int] = None
    text_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    session_id: Optional[str] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    sources: List[SourceChunk] = Field(default_factory=list)


# ------------------------------------------------------------------ #
# Summarization
# ------------------------------------------------------------------ #

class SummarizeRequest(BaseModel):
    doc_id: Optional[str] = None
    text: Optional[str] = None
    style: str = Field("concise", pattern="^(concise|detailed|bullet_points)$")
    max_length: int = Field(200, ge=50, le=2000)


class SummarizeResponse(BaseModel):
    summary: str
    style: str
    windows_processed: int


# ------------------------------------------------------------------ #
# Router
# ------------------------------------------------------------------ #

class RouteRequest(BaseModel):
    intent: Optional[str] = Field(None, description="Explicit intent; auto-detected if omitted.")
    payload: Dict[str, Any] = Field(..., description="Workflow-specific inputs.")


class RouteResponse(BaseModel):
    workflow: str
    status: str
    output: Dict[str, Any]
    steps_completed: List[str]
    errors: List[str]
    duration_seconds: float


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    status: str
    vector_store_total: int
    llm_available: bool
