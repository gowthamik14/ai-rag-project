import json

from fastapi import APIRouter, Depends

from api.schemas import (
    AuthorisedKnowledgeRequest,
    AuthorisedKnowledgeResponse,
    ChatRequest,
    ChatResponse,
)
from rag.graph import BaseRAGGraph, get_rag_graph
from services.llm_service import LLMService, get_llm_service

health_router               = APIRouter()
chat_router                 = APIRouter()
authorised_knowledge_router = APIRouter()


def _parse_rag_response(raw: str) -> tuple[str, bool, str, str]:
    """Extract (reasoning, can_be_authorised, last_updated_by, last_updated_date).

    Tries JSON first (preferred path — graph always returns JSON).
    Falls back to a plain-text VERDICT: marker for legacy/free-text responses.
    Returns (raw, False, "", "") when neither format is detected.
    """
    try:
        parsed = json.loads(raw)
        reasoning        = str(parsed.get("reasoning", raw))
        verdict          = str(parsed.get("verdict", "")).upper().strip()
        last_updated_by  = str(parsed.get("last_updated_by", ""))
        last_updated_date = str(parsed.get("last_updated_date", ""))
        return reasoning, verdict == "AUTHORISED", last_updated_by, last_updated_date
    except (json.JSONDecodeError, AttributeError, TypeError):
        verdict_marker = "VERDICT:"
        upper = raw.upper()
        if verdict_marker in upper:
            split_at = upper.rindex(verdict_marker)
            after_verdict = upper[split_at + len(verdict_marker):].strip()
            return raw, after_verdict.startswith("AUTHORISED"), "", ""
        return raw, False, "", ""


# ── /health ────────────────────────────────────────────────────────────────────

@health_router.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── /chat ──────────────────────────────────────────────────────────────────────

@chat_router.post("/chat")
def chat(
    req: ChatRequest,
    service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    reply = service.chat(req.message)
    return ChatResponse(reply=reply)


# ── /authorised-knowledge ──────────────────────────────────────────────────────

@authorised_knowledge_router.post("/authorised-knowledge")
def get_authorised_knowledge(
    req: AuthorisedKnowledgeRequest,
    rag_graph: BaseRAGGraph = Depends(get_rag_graph),
) -> AuthorisedKnowledgeResponse:
    """Run the RAG pipeline to retrieve relevant repair data and assess authorisation.

    Verdict is determined deterministically by Python (_evaluate_repair) based on
    historical status pattern and cost range.  The LLM writes a single sentence of
    reasoning to explain that verdict but never influences it.
    """
    question = (
        f"A '{req.job_title}' repair on a {req.make} {req.model} "
        f"costing £{req.job_cost:.2f} has been submitted for authorisation."
    )

    raw = rag_graph.run(
        question,
        job_title=req.job_title,
        model=req.model,
        make=req.make,
        job_cost=req.job_cost,
    )

    knowledge, can_be_authorised, last_updated_by, last_updated_date = _parse_rag_response(raw)

    return AuthorisedKnowledgeResponse(
        knowledge=knowledge,
        canbeAuthorised=can_be_authorised,
        lastUpdatedBy=last_updated_by,
        lastUpdatedDate=last_updated_date,
    )
