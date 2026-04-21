"""Golden dataset for RAG pipeline evaluation.

Each EvalCase defines:
- The input (make, model, job_title, job_cost)
- The documents the loader should return (replacing BigQuery for offline eval)
- The expected verdict (deterministic — Python decides this, not the LLM)
- Substrings that must appear in the LLM's reasoning sentence
  (limited to facts we inject into the prompt: updated-by user and date)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class EvalCase:
    name: str
    make: str
    model: str
    job_title: str
    job_cost: float
    mock_documents: list[Document]
    expected_verdict: str                          # "AUTHORISED" or "NOT AUTHORISED"
    reasoning_must_include: list[str] = field(default_factory=list)


GOLDEN_DATASET: list[EvalCase] = [
    # ── Case 1: cost in range, majority AUTHORISED ─────────────────────────────
    EvalCase(
        name="authorised_cost_in_range",
        make="Vauxhall",
        model="Crossland X",
        job_title="50000 Mile Service",
        job_cost=44_000.0,
        mock_documents=[
            Document(
                page_content="50000 mile service",
                metadata={
                    "jobLineStatus": "AUTHORISED",
                    "job_status_date": "2024-12-01",
                    "repair_cost": 45_000.0,
                    "job_status_updated_by_user": "john.smith",
                },
            ),
            Document(
                page_content="50k mile service",
                metadata={
                    "jobLineStatus": "AUTHORISED",
                    "job_status_date": "2024-06-01",
                    "repair_cost": 43_000.0,
                    "job_status_updated_by_user": "jane.doe",
                },
            ),
            Document(
                page_content="annual service 50000",
                metadata={
                    "jobLineStatus": "AUTHORISED",
                    "job_status_date": "2023-11-15",
                    "repair_cost": 46_500.0,
                    "job_status_updated_by_user": "jane.doe",
                },
            ),
        ],
        expected_verdict="AUTHORISED",
        reasoning_must_include=["john.smith", "2024-12-01"],
    ),

    # ── Case 2: cost drastically too high ──────────────────────────────────────
    EvalCase(
        name="not_authorised_cost_too_high",
        make="Vauxhall",
        model="Crossland X",
        job_title="Annual Service",
        job_cost=200_000.0,
        mock_documents=[
            Document(
                page_content="annual service",
                metadata={
                    "jobLineStatus": "AUTHORISED",
                    "job_status_date": "2024-11-15",
                    "repair_cost": 85_000.0,
                    "job_status_updated_by_user": "western.lease",
                },
            ),
        ],
        expected_verdict="NOT AUTHORISED",
        reasoning_must_include=["western.lease", "2024-11-15"],
    ),

    # ── Case 3: cost drastically too low ───────────────────────────────────────
    EvalCase(
        name="not_authorised_cost_too_low",
        make="BMW",
        model="3 Series",
        job_title="Engine Replacement",
        job_cost=1_000.0,
        mock_documents=[
            Document(
                page_content="engine replacement",
                metadata={
                    "jobLineStatus": "AUTHORISED",
                    "job_status_date": "2024-03-10",
                    "repair_cost": 85_219.0,
                    "job_status_updated_by_user": "sarah.connor",
                },
            ),
        ],
        expected_verdict="NOT AUTHORISED",
        reasoning_must_include=["sarah.connor", "2024-03-10"],
    ),

    # ── Case 4: majority of historical records DECLINED ────────────────────────
    EvalCase(
        name="not_authorised_majority_declined",
        make="Ford",
        model="Focus",
        job_title="Brake Replacement",
        job_cost=500.0,
        mock_documents=[
            Document(
                page_content="brake replacement",
                metadata={
                    "jobLineStatus": "DECLINED",
                    "job_status_date": "2024-12-01",
                    "repair_cost": 480.0,
                    "job_status_updated_by_user": "mike.jones",
                },
            ),
            Document(
                page_content="brake replacement",
                metadata={
                    "jobLineStatus": "DECLINED",
                    "job_status_date": "2024-09-01",
                    "repair_cost": 510.0,
                    "job_status_updated_by_user": "mike.jones",
                },
            ),
            Document(
                page_content="brake replacement",
                metadata={
                    "jobLineStatus": "DECLINED",
                    "job_status_date": "2024-06-01",
                    "repair_cost": 495.0,
                    "job_status_updated_by_user": "mike.jones",
                },
            ),
            Document(
                page_content="brake replacement",
                metadata={
                    "jobLineStatus": "AUTHORISED",
                    "job_status_date": "2024-01-01",
                    "repair_cost": 490.0,
                    "job_status_updated_by_user": "old.user",
                },
            ),
        ],
        expected_verdict="NOT AUTHORISED",
        # Attribution comes from the most-recent record (sorted by date)
        reasoning_must_include=["mike.jones", "2024-12-01"],
    ),

    # ── Case 5: no historical records at all ───────────────────────────────────
    EvalCase(
        name="not_authorised_no_records",
        make="Toyota",
        model="Corolla",
        job_title="Transmission Overhaul",
        job_cost=3_000.0,
        mock_documents=[],
        expected_verdict="NOT AUTHORISED",
        reasoning_must_include=[],  # short-circuit path — no LLM call
    ),
]
