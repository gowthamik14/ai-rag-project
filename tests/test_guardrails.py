"""
Unit tests for TopicGuardrail — no LLM or live services needed.
LLM classification is mocked so these run fully offline.
"""
from unittest.mock import MagicMock
import pytest

from core.guardrails import TopicGuardrail
from core.llm import OllamaLLM


def _guardrail(llm_response: str = "yes") -> TopicGuardrail:
    """Return a guardrail with a mocked LLM."""
    llm = MagicMock(spec=OllamaLLM)
    llm.generate.return_value = llm_response
    return TopicGuardrail(llm=llm)


# ------------------------------------------------------------------ #
# Keyword allow
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("question", [
    "What is the tire pressure for a Honda Civic?",
    "My engine warning light is on",
    "How often should I change the oil in my Toyota?",
    "What does the ABS light mean?",
    "BMW 3 series brake pad replacement",
    "EV battery range on a Tesla Model 3",
])
def test_vehicle_questions_allowed_by_keyword(question):
    g = _guardrail()
    allowed, reason = g.check(question)
    assert allowed is True
    assert reason == "keyword_match"
    # LLM should NOT be called for keyword matches
    g._llm.generate.assert_not_called()


# ------------------------------------------------------------------ #
# Keyword block
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("question", [
    "Give me a recipe for chocolate cake",
    "Write me a poem about the ocean",
    "What is the weather forecast for tomorrow?",
    "Who won the election?",
    "Translate this sentence to French",
])
def test_off_topic_questions_blocked_by_keyword(question):
    g = _guardrail()
    allowed, reason = g.check(question)
    assert allowed is False
    assert reason == "keyword_block"
    g._llm.generate.assert_not_called()


# ------------------------------------------------------------------ #
# LLM classification — ambiguous questions
# ------------------------------------------------------------------ #

def test_ambiguous_vehicle_question_allowed_by_llm():
    g = _guardrail(llm_response="yes")
    allowed, reason = g.check("What does that warning light on the dashboard mean?")
    assert allowed is True
    assert reason == "llm_classification"
    g._llm.generate.assert_called_once()


def test_ambiguous_non_vehicle_question_blocked_by_llm():
    g = _guardrail(llm_response="no")
    allowed, reason = g.check("Can you help me with my homework?")
    assert allowed is False
    assert reason == "llm_block"


def test_llm_error_defaults_to_block():
    llm = MagicMock(spec=OllamaLLM)
    llm.generate.side_effect = ConnectionError("Ollama not running")
    g = TopicGuardrail(llm=llm)
    allowed, reason = g.check("What is the recommended tyre rotation interval?")
    # keyword_match should catch this before LLM — adjust to a truly ambiguous question
    allowed, reason = g.check("Can you help me understand this warning?")
    assert allowed is False
    assert reason == "llm_error"


# ------------------------------------------------------------------ #
# Blocked response message
# ------------------------------------------------------------------ #

def test_blocked_response_message_is_set():
    assert "vehicle" in TopicGuardrail.BLOCKED_RESPONSE.lower()


# ------------------------------------------------------------------ #
# is_allowed helper
# ------------------------------------------------------------------ #

def test_is_allowed_returns_bool():
    g = _guardrail()
    assert g.is_allowed("What oil filter fits a Ford Focus?") is True
    assert g.is_allowed("Write me a poem") is False
