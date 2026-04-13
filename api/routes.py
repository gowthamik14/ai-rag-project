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

    The LLM is asked to end its response with VERDICT: AUTHORISED or
    VERDICT: NOT AUTHORISED so the result can be parsed reliably.
    """
    question = (
        f"A '{req.job_title}' repair on a {req.make} {req.model} "
        f"costing £{req.job_cost:.2f} has been submitted for authorisation."
    )

    knowledge = rag_graph.run(
        question,
        job_title=req.job_title,
        model=req.model,
        make=req.make,
    )

    try:
        parsed = json.loads(knowledge)
        knowledge = str(parsed.get("reasoning", knowledge))
        verdict = str(parsed.get("verdict", "")).upper().strip()
        can_be_authorised = verdict == "AUTHORISED"
    except (json.JSONDecodeError, AttributeError, TypeError):
        # Fallback: look for VERDICT: marker in free-text responses.
        verdict_marker = "VERDICT:"
        upper = knowledge.upper()
        if verdict_marker in upper:
            split_at = upper.rindex(verdict_marker)
            after_verdict = upper[split_at + len(verdict_marker):].strip()
            can_be_authorised = after_verdict.startswith("AUTHORISED")
        else:
            can_be_authorised = False

    return AuthorisedKnowledgeResponse(
        knowledge=knowledge,
        canbeAuthorised=can_be_authorised,
    )
