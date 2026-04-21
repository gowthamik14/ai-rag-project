"""RAG pipeline evaluation runner.

Evaluation layers
-----------------
1. Verdict Accuracy         — always runs, no external services needed.
                              Checks that _evaluate_repair() produces the
                              correct AUTHORISED / NOT AUTHORISED verdict.

2. Reasoning Completeness   — always runs, no external services needed.
                              Checks that the LLM's one-sentence reasoning
                              includes attribution facts we injected into the
                              prompt (updated-by user, date).

3. Retrieval Relevance      — always runs, no external services needed.
                              Uses real HuggingFace embeddings + FAISS to
                              verify the retrieve node returns semantically
                              relevant chunks for each job title.

4. DeepEval Faithfulness    — optional (--deepeval flag), requires Ollama.
                              Uses qwen2.5:14b as the judge to score whether
                              the reasoning stays faithful to the retrieved
                              context (no hallucinated facts).

Usage
-----
    # Deterministic metrics only (offline, fast):
    python -m eval.run_eval

    # Include DeepEval faithfulness (requires Ollama running with qwen2.5:14b):
    python -m eval.run_eval --deepeval

    # Save results to a custom path:
    python -m eval.run_eval --output results/eval_$(date +%Y%m%d).json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from eval.golden_dataset import GOLDEN_DATASET, EvalCase
from rag.document_loader import DocumentLoader
from rag.graph import RAGGraph


# ── infrastructure helpers ────────────────────────────────────────────────────

class _StaticLoader(DocumentLoader):
    """DocumentLoader that always returns a fixed list of documents."""

    def __init__(self, docs: List[Document]) -> None:
        self._docs = docs

    def load(self) -> List[Document]:
        return self._docs


def _make_graph(
    docs: List[Document],
    llm_service: Any = None,
    use_real_embeddings: bool = False,
) -> RAGGraph:
    """Build a RAGGraph backed by *docs*, bypassing BigQuery.

    By default uses FakeEmbeddings (fast, no model download) so the
    deterministic and reasoning-completeness metrics can run offline.
    Pass use_real_embeddings=True for the RAGAS evaluation which needs
    real semantic similarity to measure context precision/recall.
    """
    if use_real_embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings
        from config.settings import settings
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        embeddings = _FakeEmbeddings()

    kwargs: Dict[str, Any] = dict(
        loader_factory=lambda m, mk, jc: _StaticLoader(docs),
        embeddings=embeddings,
    )
    if llm_service is not None:
        kwargs["llm_service"] = llm_service
    return RAGGraph(**kwargs)


class _FakeEmbeddings(Embeddings):
    """Minimal Embeddings stub — all vectors identical, so every chunk passes
    the similarity filter.  Fast and dependency-free."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 8


class _CapturingLLM:
    """Thin LLM wrapper that records the last prompt and delegates to a real service."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_prompt: Optional[str] = None
        self.last_response: Optional[str] = None

    def chat(self, message: str) -> str:
        self.last_prompt = message
        self.last_response = self._inner.chat(message)
        return self.last_response


def _run_case(case: EvalCase, llm_service: Any = None) -> Dict[str, Any]:
    graph = _make_graph(case.mock_documents, llm_service)
    raw = graph.run(
        question=f"Evaluate repair: {case.job_title}",
        job_title=case.job_title,
        model=case.model,
        make=case.make,
        job_cost=case.job_cost,
    )
    return json.loads(raw)


# ── metric 1: verdict accuracy (no LLM judge) ────────────────────────────────

def evaluate_verdict_accuracy() -> Dict[str, Any]:
    """Checks Python's _evaluate_repair() returns the expected verdict."""
    results = []
    for case in GOLDEN_DATASET:
        output = _run_case(case)
        actual = output.get("verdict", "")
        passed = actual == case.expected_verdict
        results.append({
            "case": case.name,
            "expected": case.expected_verdict,
            "actual": actual,
            "pass": passed,
        })
    score = sum(1 for r in results if r["pass"]) / len(results)
    return {"metric": "VerdictAccuracy", "score": score, "details": results}


# ── metric 2: reasoning completeness (no LLM judge) ──────────────────────────

def evaluate_reasoning_completeness() -> Dict[str, Any]:
    """Checks the LLM's reasoning sentence includes required attribution facts.

    Only verifies facts we inject into the prompt (updated-by user, date).
    Does NOT ask the LLM to evaluate itself — purely string matching.
    """
    results = []
    for case in GOLDEN_DATASET:
        if not case.reasoning_must_include:
            # Short-circuit cases (no documents) never call the LLM — skip.
            results.append({"case": case.name, "pass": True, "missing": [], "skipped": True})
            continue
        output = _run_case(case)
        reasoning = output.get("reasoning", "").lower()
        missing = [s for s in case.reasoning_must_include if s.lower() not in reasoning]
        results.append({
            "case": case.name,
            "reasoning": output.get("reasoning", ""),
            "pass": len(missing) == 0,
            "missing": missing,
        })
    applicable = [r for r in results if not r.get("skipped")]
    score = sum(1 for r in applicable if r["pass"]) / len(applicable) if applicable else 1.0
    return {"metric": "ReasoningCompleteness", "score": score, "details": results}


# ── metric 3: DeepEval faithfulness (requires Ollama) ────────────────────────

def evaluate_deepeval_faithfulness() -> Dict[str, Any]:
    """Asks qwen2.5:14b to judge whether the LLM's reasoning stays faithful
    to the retrieved context.  Catches hallucinated cost figures or invented
    repair descriptions that were not in the historical records.
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase
            from deepeval.models.base_model import DeepEvalBaseLLM
            from deepeval import evaluate as deepeval_evaluate
    except ImportError as exc:
        return {"metric": "DeepEval/Faithfulness", "error": f"deepeval not installed: {exc}", "score": None}

    try:
        from langchain_ollama import OllamaLLM
        from config.settings import settings
    except ImportError as exc:
        return {"metric": "DeepEval/Faithfulness", "error": str(exc), "score": None}

    # ── Ollama-backed judge for deepeval ──────────────────────────────────────
    class _OllamaJudge(DeepEvalBaseLLM):
        def load_model(self) -> OllamaLLM:
            return OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)

        def generate(self, prompt: str) -> str:
            return self.load_model().invoke(prompt)

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return settings.llm_model

    judge = _OllamaJudge()
    metric = FaithfulnessMetric(
        threshold=0.7,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    test_cases = []
    for case in GOLDEN_DATASET:
        if not case.mock_documents:
            continue  # no-records cases don't call the LLM — nothing to judge
        output = _run_case(case)
        # Build context exactly as the graph does — one line per record.
        context_lines = [
            f"{doc.page_content} | status: {doc.metadata.get('jobLineStatus')} "
            f"| cost: £{float(doc.metadata.get('repair_cost', 0)):.2f} "
            f"| date: {doc.metadata.get('job_status_date')} "
            f"| updated by: {doc.metadata.get('job_status_updated_by_user')}"
            for doc in case.mock_documents
        ]
        test_cases.append(
            LLMTestCase(
                input=(
                    f"{case.job_title} for {case.make} {case.model}, "
                    f"submitted cost £{case.job_cost:.2f}"
                ),
                actual_output=output.get("reasoning", ""),
                retrieval_context=context_lines,
            )
        )

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_result = deepeval_evaluate(
                test_cases, [metric], run_async=False, print_results=False
            )
        scores = [r.metrics[0].score for r in eval_result.test_results]
        reasons = [r.metrics[0].reason for r in eval_result.test_results]
    except Exception as exc:
        return {"metric": "DeepEval/Faithfulness", "error": str(exc), "score": None}

    avg = sum(scores) / len(scores) if scores else 0.0
    cases_with_docs = [c for c in GOLDEN_DATASET if c.mock_documents]
    details = [
        {
            "case": cases_with_docs[i].name,
            "score": scores[i],
            "reason": reasons[i],
            "pass": scores[i] >= 0.7,
        }
        for i in range(len(scores))
    ]
    return {"metric": "DeepEval/Faithfulness", "score": avg, "details": details}


# ── metric 4: retrieval relevance (real embeddings, no external LLM) ─────────

def evaluate_retrieval_relevance() -> Dict[str, Any]:
    """Verifies the retrieve node returns semantically relevant chunks.

    Uses the same HuggingFace all-MiniLM-L6-v2 embeddings + FAISS index
    that the production pipeline uses — so this tests end-to-end retrieval
    quality, not fake vector stores.

    For each case we check:
    - At least one chunk is retrieved (coverage).
    - All retrieved chunks belong to the same repair type as the query
      (precision), measured by checking that a keyword from the job title
      appears in the chunk text OR status metadata.

    No LLM judge is needed — purely embedding + string checks.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from config.settings import settings
    except ImportError as exc:
        return {"metric": "RetrievalRelevance", "error": str(exc), "score": None}

    results = []
    for case in GOLDEN_DATASET:
        if not case.mock_documents:
            results.append({
                "case": case.name,
                "pass": True,
                "skipped": True,
                "note": "no documents — retrieve node short-circuits",
            })
            continue

        graph = _make_graph(case.mock_documents, use_real_embeddings=True)
        retrieve_result = graph._retrieve({  # type: ignore[attr-defined]
            "question":        case.job_title,
            "job_title":       case.job_title,
            "job_cost":        case.job_cost,
            "make":            case.make,
            "model":           case.model,
            "documents":       case.mock_documents,
            "relevant_chunks": [],
            "answer":          "",
        })
        chunks = retrieve_result.get("relevant_chunks", [])

        # Coverage: at least one chunk returned.
        if not chunks:
            results.append({
                "case": case.name,
                "pass": False,
                "retrieved": 0,
                "note": "no chunks retrieved — retriever filtered everything out",
            })
            continue

        # Precision: each retrieved chunk should share at least one keyword
        # with the job title (case-insensitive, partial match on any word ≥ 4 chars).
        title_words = {w.lower() for w in case.job_title.split() if len(w) >= 4}
        irrelevant = [
            chunk.page_content
            for chunk in chunks
            if not any(w in chunk.page_content.lower() for w in title_words)
            and not any(w in (chunk.metadata.get("jobLineStatus") or "").lower() for w in title_words)
        ]
        precision = 1.0 - (len(irrelevant) / len(chunks))
        passed = precision >= 0.5 and len(chunks) > 0

        results.append({
            "case": case.name,
            "pass": passed,
            "retrieved": len(chunks),
            "precision": round(precision, 2),
            "irrelevant_chunks": irrelevant[:2],  # show at most 2 for brevity
        })

    applicable = [r for r in results if not r.get("skipped")]
    score = sum(1 for r in applicable if r["pass"]) / len(applicable) if applicable else 1.0
    return {"metric": "RetrievalRelevance", "score": score, "details": results}


# ── reporting ─────────────────────────────────────────────────────────────────

def _print_result(result: Dict[str, Any]) -> None:
    metric = result.get("metric", "?")
    score = result.get("score")
    error = result.get("error")

    if error:
        print(f"  {metric}: ERROR — {error}")
        return

    pct = f"{score * 100:.0f}%" if score is not None else "?"
    status = "PASS" if (score or 0) >= 0.8 else "FAIL"
    print(f"  {metric}: {pct} [{status}]")
    for d in result.get("details", []):
        icon = "✓" if d.get("pass") else "✗"
        line = f"    {icon} {d.get('case', '?')}"
        if d.get("skipped"):
            line += " (skipped — no LLM call)"
        elif not d.get("pass"):
            if d.get("missing"):
                line += f"  — missing: {d['missing']}"
            if "expected" in d:
                line += f"  — expected: {d['expected']}, got: {d.get('actual', '?')}"
        if "reason" in d and d.get("reason"):
            line += f"\n      judge: {d['reason']}"
        print(line)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--deepeval", action="store_true",
        help="Run DeepEval FaithfulnessMetric (requires Ollama with qwen2.5:14b)"
    )
    parser.add_argument(
        "--output", default="eval_results.json",
        help="JSON file to write results to (default: eval_results.json)"
    )
    args = parser.parse_args()

    all_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 60)
    print("RAG Pipeline Evaluation")
    print("=" * 60)

    print("\n[1/4] Verdict Accuracy (deterministic, no LLM judge)")
    r1 = evaluate_verdict_accuracy()
    _print_result(r1)
    all_results.append(r1)

    print("\n[2/4] Reasoning Completeness (attribution check, no LLM judge)")
    r2 = evaluate_reasoning_completeness()
    _print_result(r2)
    all_results.append(r2)

    print("\n[3/4] Retrieval Relevance (real HuggingFace embeddings + FAISS)")
    r3 = evaluate_retrieval_relevance()
    _print_result(r3)
    all_results.append(r3)

    if args.deepeval:
        print("\n[4/4] DeepEval Faithfulness (Ollama judge — slow, ~30s per case)")
        r4 = evaluate_deepeval_faithfulness()
        _print_result(r4)
        all_results.append(r4)
    else:
        print("\n[4/4] DeepEval Faithfulness  — skipped (pass --deepeval to enable)")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = all(
        r.get("score", 0) >= 0.8
        for r in all_results
        if r.get("score") is not None
    )
    for r in all_results:
        score = r.get("score")
        if score is None:
            status = "ERROR" if r.get("error") else "N/A"
        else:
            status = "PASS" if score >= 0.8 else "FAIL"
        print(f"  {r['metric']:<35} {status}")
    print()

    with open(args.output, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Full results written to: {args.output}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
