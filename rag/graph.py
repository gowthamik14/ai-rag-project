from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from config.settings import settings
from rag.document_loader import DocumentLoader, create_model_loader
from utils.logger import get_logger

if TYPE_CHECKING:
    from services.llm_service import LLMService


_log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum L2 distance (vs the best-match score) a chunk may have and still be
# considered "similar enough" to be included in the LLM context.
# FAISS with normalised embeddings gives L2 distances in [0, 2].  A buffer of
# 0.3 admits paraphrases of the same repair type while excluding unrelated
# records (e.g. a test entry "1123") that happen to have a recent date.
_SIMILARITY_SCORE_BUFFER: float = 0.3

# Maximum number of (make, model, job_cost) entries to keep in the per-instance
# FAISS vector-store cache.  Least-recently-built entry is evicted when full.
_VS_CACHE_SIZE: int = 32

# Prompt given to the LLM with all retrieved historical records.
# The verdict is already determined by Python (_evaluate_repair) before this
# prompt is constructed.  The LLM's sole job is to write a single sentence
# of reasoning that references the historical context to explain that verdict.
_GENERATE_PROMPT = """\
You are explaining a vehicle repair decision.

Verdict: {verdict}

Facts:
- Vehicle: {make} {model}
- Submitted repair: {job_title}
- Submitted cost: £{job_cost:.2f}
- Cost assessment: {cost_assessment}
- Updated by: {last_updated_by}
- Date: {last_updated_date}

Historical records (most recent first):
{context}

Rules:
- Do NOT change the verdict
- Do NOT add information not present in the facts above
- Use exactly one sentence
- Include who last updated this repair and the date in your sentence

Return JSON only:
{{"reasoning": "..."}}\
"""

# Extracts the first {...} block from a response that contains surrounding prose.
# Uses a greedy outer-brace approach so it captures the outermost JSON object
# even when it contains nested objects (e.g. {"key": {"inner": "val"}}).
_JSON_EXTRACT_RE = re.compile(r"\{.*\}", re.DOTALL)


# ── Graph state ────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """Data that flows between nodes in the RAG graph."""
    question:        str             # full request shown to the LLM
    job_title:       str             # bare repair description — used as the FAISS query
    job_cost:        float           # submitted repair cost in GBP
    make:            str             # vehicle make  — used by _load to filter BigQuery
    model:           str             # vehicle model — used by _load to filter BigQuery
    documents:       list[Document]  # raw rows from BigQuery
    relevant_chunks: list[Document]  # top-k chunks after vector search
    answer:          str


# ── Abstraction ────────────────────────────────────────────────────────────────

class BaseRAGGraph(ABC):
    """Abstract contract for the RAG pipeline."""

    @abstractmethod
    def run(self, question: str, job_title: str, model: str, make: str, job_cost: float) -> str:
        """Run the pipeline for a given question, filtering data by vehicle make/model."""


# ── Embeddings singleton ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse across requests."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Concrete implementation ────────────────────────────────────────────────────

class RAGGraph(BaseRAGGraph):
    """LangGraph-based RAG pipeline.

    Graph structure:
        load → retrieve → generate → END

    - load     : fetches Documents from BigQuery filtered by vehicle model and make.
    - retrieve : splits into chunks, builds an in-memory FAISS index, returns
                 the top-k most relevant chunks for the question.
    - generate : Python (_evaluate_repair) determines the verdict deterministically
                 (AUTHORISED / NOT AUTHORISED) from status history and cost range.
                 The LLM then writes a single sentence of reasoning to explain that
                 verdict using the historical context.  Short-circuits to
                 NOT AUTHORISED without an LLM call when no similar records were found.

    Args:
        loader_factory:       Callable that takes (model, make, job_cost) and returns a
                              DocumentLoader. Defaults to create_model_loader.
                              Inject a fake in tests to avoid BigQuery calls.
        embeddings:           LangChain Embeddings implementation. Defaults to
                              HuggingFaceEmbeddings. Inject a fake in tests to
                              avoid downloading the sentence-transformers model.
        vector_store_factory: Callable that takes (chunks, embeddings) and
                              returns a vector store with a similarity_search
                              method. Defaults to FAISS.from_documents.
                              Inject a fake in tests to avoid native FAISS calls.
        llm_service:          LLMService used by the generate node. Defaults to the
                              application-wide singleton from get_llm_service().
                              Inject a FakeLLMService in tests to avoid Ollama calls.
    """

    def __init__(
        self,
        loader_factory: Callable[[str, str, float], DocumentLoader] = create_model_loader,
        embeddings: Embeddings | None = None,
        vector_store_factory: Callable[[list[Document], Embeddings], Any] | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        from services.llm_service import get_llm_service  # local import avoids circular dep

        self._loader_factory        = loader_factory
        self._embeddings            = embeddings or _get_embeddings()
        self._vector_store_factory  = vector_store_factory or FAISS.from_documents
        self._llm_service           = llm_service or get_llm_service()
        self._splitter              = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        # Bounded LRU cache: reuse FAISS indexes for repeated (make, model, job_cost)
        # combinations to avoid re-embedding on every request.
        # Trade-off: if BigQuery data changes between requests with the same key,
        # the cached index will be stale.  Acceptable for the current use-case where
        # the same repair/cost combination is unlikely to be re-evaluated against
        # materially different historical data within a single process lifetime.
        # Note: not thread-safe; acceptable for the current single-worker deployment.
        self._vs_cache: OrderedDict[tuple[str, str, float], Any] = OrderedDict()
        self._graph = self._build_graph()

    # ── public ────────────────────────────────────────────────────────────

    def run(self, question: str, job_title: str, model: str, make: str, job_cost: float) -> str:
        """Run the full RAG pipeline and return the generated answer."""
        result = self._graph.invoke({
            "question":        question,
            "job_title":       job_title,
            "job_cost":        job_cost,
            "make":            make,
            "model":           model,
            "documents":       [],
            "relevant_chunks": [],
            "answer":          "",
        })
        return result["answer"]

    # ── graph construction ─────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        builder = StateGraph(RAGState)
        builder.add_node("load",     self._load)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.set_entry_point("load")
        builder.add_edge("load",     "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    # ── nodes ──────────────────────────────────────────────────────────────

    def _load(self, state: RAGState) -> dict:
        """Node: fetch Documents from BigQuery filtered by vehicle make and model."""
        _log.info(
            "load: querying BigQuery",
            extra={"make": state["make"], "model": state["model"], "job_cost": state["job_cost"]},
        )
        loader = self._loader_factory(state["model"], state["make"])
        docs = loader.load()
        _log.info(
            "load: fetched documents",
            extra={"make": state["make"], "model": state["model"], "document_count": len(docs)},
        )
        return {"documents": docs}

    def _retrieve(self, state: RAGState) -> dict:
        """Node: chunk documents, build in-memory FAISS, return top-k relevant chunks.

        Uses similarity scores to discard records that are semantically unrelated
        to job_title (e.g. a test entry "1123" in a sparse dataset).  Only chunks
        whose L2 distance is within _SIMILARITY_SCORE_BUFFER of the best match are
        kept, so the LLM context only contains like-for-like repairs.
        """
        if not state["documents"]:
            _log.warning("retrieve: no documents loaded — skipping vector search",
                         extra={"make": state["make"], "model": state["model"]})
            return {"relevant_chunks": []}

        chunks = self._splitter.split_documents(state["documents"])
        if not chunks:
            _log.warning("retrieve: document splitting produced no chunks",
                         extra={"make": state["make"], "model": state["model"]})
            return {"relevant_chunks": []}

        _log.info(
            "retrieve: built chunks",
            extra={"chunk_count": len(chunks), "job_title": state["job_title"]},
        )

        cache_key = (state["make"].lower(), state["model"].lower())
        cache_hit = cache_key in self._vs_cache
        if not cache_hit:
            self._vs_cache[cache_key] = self._vector_store_factory(chunks, self._embeddings)
            if len(self._vs_cache) > _VS_CACHE_SIZE:
                self._vs_cache.popitem(last=False)  # evict oldest entry
        _log.debug("retrieve: vector store", extra={"cache_hit": cache_hit})

        vector_store = self._vs_cache[cache_key]
        # Use job_title (bare repair phrase) as the retrieval query — it matches
        # repair_description embeddings far better than the full verbose question.
        results: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
            state["job_title"], k=settings.retrieval_top_k
        )
        if not results:
            _log.warning("retrieve: similarity search returned no results",
                         extra={"job_title": state["job_title"]})
            return {"relevant_chunks": []}

        best_score = results[0][1]
        relevant = [
            doc for doc, score in results
            if score <= best_score + _SIMILARITY_SCORE_BUFFER
        ]
        _log.info(
            "retrieve: similarity search complete",
            extra={
                "job_title":        state["job_title"],
                "candidates":       len(results),
                "passed_filter":    len(relevant),
                "best_score":       round(best_score, 4),
                "score_threshold":  round(best_score + _SIMILARITY_SCORE_BUFFER, 4),
            },
        )
        return {"relevant_chunks": relevant}

    def _generate(self, state: RAGState) -> dict:
        """Node: evaluate the repair deterministically, then ask the LLM to explain.

        Round 2 architecture — two responsibilities, clearly separated:

        1. Python (_evaluate_repair) decides the verdict deterministically based
           on the status majority of historical records and a cost-range check.
           The LLM never influences the verdict.

        2. The LLM is given the pre-computed verdict and all historical records
           and writes a single sentence of reasoning that references those facts.

        Short-circuits to NOT AUTHORISED (no LLM call) when no similar records
        were found — there is nothing for the model to explain.

        Attribution fields (last_updated_by, last_updated_date) are taken from
        the most recent record's metadata.
        """
        if not state["relevant_chunks"]:
            _log.warning(
                "generate: no relevant chunks — short-circuiting to NOT AUTHORISED",
                extra={"make": state["make"], "model": state["model"], "job_title": state["job_title"]},
            )
            return {"answer": json.dumps({
                "reasoning": "No similar past repair records found for this vehicle.",
                "verdict": "NOT AUTHORISED",
                "last_updated_by": "",
                "last_updated_date": "",
            })}

        # Sort most-recent-first so the LLM sees the strongest precedent at the top.
        sorted_chunks = sorted(
            state["relevant_chunks"],
            key=lambda d: _to_iso_date_str(d.metadata.get("job_status_date")),
            reverse=True,
        )

        # Attribution always comes from the most recent record.
        latest_meta   = sorted_chunks[0].metadata
        last_updated_by   = str(latest_meta.get("job_status_updated_by_user") or "")
        last_updated_date = _to_iso_date_str(latest_meta.get("job_status_date")) or ""

        # Build context — every retrieved record is shown to the LLM.
        costs: list[float] = []
        statuses: list[str] = []
        context_lines: list[str] = []
        for i, chunk in enumerate(sorted_chunks, 1):
            m        = chunk.metadata
            status   = m.get("jobLineStatus") or "unknown"
            cost     = m.get("repair_cost")
            cost_str = f"£{float(cost):.2f}" if cost is not None else "not recorded"
            date_str = _to_iso_date_str(m.get("job_status_date")) or "unknown"
            user     = m.get("job_status_updated_by_user") or ""
            by_part  = f", updated by {user}" if user else ""
            context_lines.append(
                f"{i}. {chunk.page_content} | status: {status} | "
                f"date: {date_str} | cost: {cost_str}{by_part}"
            )
            if cost is not None:
                costs.append(float(cost))
            statuses.append(status)

        # Python evaluates both status history and cost — results are passed as
        # plain-English facts so the LLM never has to compare numbers or tally
        # verdicts itself.
        cost_assessment, verdict = _evaluate_repair(state["job_cost"], costs, statuses)

        _log.info(
            "generate: verdict determined",
            extra={
                "make":            state["make"],
                "model":           state["model"],
                "job_title":       state["job_title"],
                "job_cost":        state["job_cost"],
                "verdict":         verdict,
                "cost_assessment": cost_assessment,
                "context_records": len(sorted_chunks),
            },
        )

        prompt = _GENERATE_PROMPT.format(
            make=state["make"],
            model=state["model"],
            job_title=state["job_title"],
            job_cost=state["job_cost"],
            context="\n".join(context_lines),
            cost_assessment=cost_assessment,
            verdict=verdict,
            last_updated_by=last_updated_by or "unknown",
            last_updated_date=last_updated_date or "unknown",
        )

        _log.debug("generate: sending prompt to LLM", extra={"prompt": prompt})
        raw = self._llm_service.chat(prompt)
        _log.debug("generate: received LLM response", extra={"raw_response": raw})

        obj = _parse_llm_response(raw)
        reasoning = str(obj.get("reasoning", "")).strip()

        if not reasoning:
            _log.warning("generate: LLM returned empty reasoning — using fallback",
                         extra={"raw_response": raw})
            reasoning = "Unable to determine reasoning from historical records."

        _log.info(
            "generate: final answer",
            extra={"verdict": verdict, "last_updated_by": last_updated_by, "last_updated_date": last_updated_date},
        )

        return {"answer": json.dumps({
            "reasoning":         reasoning,
            "verdict":           verdict,
            "last_updated_by":   last_updated_by,
            "last_updated_date": last_updated_date,
        })}


# ── helpers ────────────────────────────────────────────────────────────────────

_DECLINED_STATUSES: frozenset[str] = frozenset({"DECLINED", "NOT AUTHORISED", "REJECTED"})


def _evaluate_repair(
    job_cost: float,
    historical_costs: list[float],
    historical_statuses: list[str],
) -> tuple[str, str]:
    """Evaluate the repair against historical status pattern and cost.

    Checks in order:
    1. If the majority of historical records were declined → NOT AUTHORISED.
    2. If the submitted cost is significantly out of range → NOT AUTHORISED.
    3. Otherwise → AUTHORISED.

    Returns ("plain-English assessment", "AUTHORISED" | "NOT AUTHORISED").
    """
    # ── Status pattern check ───────────────────────────────────────────────────
    if historical_statuses:
        declined_count = sum(
            1 for s in historical_statuses if s.upper() in _DECLINED_STATUSES
        )
        total = len(historical_statuses)
        if declined_count > total / 2:
            assessment = (
                f"{declined_count} of {total} historical records for this repair "
                f"were declined."
            )
            return assessment, "NOT AUTHORISED"

    # ── Cost range check ───────────────────────────────────────────────────────
    if not historical_costs:
        return "No historical cost data available to compare against.", "NOT AUTHORISED"

    avg_cost = sum(historical_costs) / len(historical_costs)

    if job_cost > avg_cost * 2:
        assessment = (
            f"The submitted cost of £{job_cost:.2f} is significantly higher than "
            f"the historical average of £{avg_cost:.2f}."
        )
        return assessment, "NOT AUTHORISED"

    if job_cost < avg_cost / 2:
        assessment = (
            f"The submitted cost of £{job_cost:.2f} is significantly lower than "
            f"the historical average of £{avg_cost:.2f}."
        )
        return assessment, "NOT AUTHORISED"

    assessment = (
        f"The submitted cost of £{job_cost:.2f} is in line with "
        f"the historical average of £{avg_cost:.2f}."
    )
    return assessment, "AUTHORISED"


def _parse_llm_response(raw: str) -> dict:
    """Extract a dict from a raw LLM response.

    Tries in order:
    1. Direct JSON parse (ideal — Ollama format="json" usually delivers this).
    2. Regex extraction of the first {...} block (handles surrounding prose).
    3. Empty dict fallback so the caller always gets a usable dict.
    """
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    match = _JSON_EXTRACT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    return {}


def _to_iso_date_str(value: Any) -> str:
    """Return an ISO-8601 date string (YYYY-MM-DD) suitable for lexicographic sorting.

    Handles the three forms BigQuery can produce:
    - datetime.date / datetime.datetime objects  → extract date part, return YYYY-MM-DD
    - ISO strings ("2024-03-15")                 → pass through
    - Common non-ISO strings ("15/03/2024")      → reformat to ISO
    - None / anything else                       → "" (sorts earliest)
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value:
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                continue
        return value  # already ISO or unknown — return as-is
    return ""


# ── dependency ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rag_graph() -> BaseRAGGraph:
    """FastAPI dependency — returns a single compiled RAGGraph instance."""
    return RAGGraph()
