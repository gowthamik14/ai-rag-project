"""
TopicGuardrail — restricts the LLM to vehicle-related questions only.

Two-layer check (fast → accurate):

  Layer 1 — Keyword allow-list
    If the question contains any known vehicle keyword → allowed immediately.
    If it contains a known off-topic signal (e.g. "recipe", "poem") → blocked immediately.

  Layer 2 — LLM classification  (only for ambiguous questions)
    Asks Qwen to classify the question as "vehicle" or "other".
    Runs entirely offline — no internet connection used.

Usage
-----
    guardrail = TopicGuardrail()

    allowed, reason = guardrail.check("What is the tire pressure for a Honda Civic?")
    # → (True, "keyword_match")

    allowed, reason = guardrail.check("Write me a poem about the ocean")
    # → (False, "keyword_block")

    allowed, reason = guardrail.check("What does the warning light mean?")
    # → (True, "llm_classification")
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Keyword lists
# ------------------------------------------------------------------ #

VEHICLE_KEYWORDS = {
    # General
    "vehicle", "vehicles", "car", "cars", "truck", "trucks", "van", "vans",
    "suv", "bus", "buses", "motorcycle", "motorcycles", "bike", "scooter",
    "fleet", "automobile", "automotive", "motor",
    # Parts & systems
    "engine", "transmission", "gearbox", "brake", "brakes", "braking",
    "tyre", "tyres", "tire", "tires", "wheel", "wheels", "axle", "suspension",
    "steering", "exhaust", "battery", "alternator", "starter", "radiator",
    "coolant", "oil", "fuel", "diesel", "petrol", "gasoline", "hybrid",
    "electric vehicle", "ev", "hybrid vehicle", "turbo", "clutch",
    "differential", "drivetrain", "powertrain", "catalytic", "muffler",
    "headlight", "taillight", "windshield", "wiper", "airbag", "seatbelt",
    "odometer", "speedometer", "dashboard", "instrument panel",
    # Maintenance & service
    "service", "servicing", "maintenance", "repair", "workshop", "mechanic",
    "diagnostic", "fault code", "error code", "obd", "mot", "inspection",
    "recall", "warranty", "overhaul", "tune up", "tune-up",
    "oil change", "filter", "spark plug", "timing belt", "cam belt",
    # Driving & operations
    "drive", "driving", "mileage", "mpg", "fuel economy", "range",
    "acceleration", "horsepower", "torque", "rpm", "gear", "gears",
    "cruise control", "traction", "abs", "esp", "parking", "reversing",
    # Makes / common models (expand as needed)
    "honda", "toyota", "ford", "bmw", "mercedes", "audi", "volkswagen",
    "vw", "hyundai", "kia", "nissan", "mazda", "subaru", "volvo",
    "peugeot", "renault", "fiat", "tesla", "chevrolet", "chevy", "jeep",
    "ram", "dodge", "chrysler", "lexus", "infiniti", "acura", "cadillac",
    "buick", "gmc", "land rover", "jaguar", "porsche", "ferrari",
    "lamborghini", "maserati", "bentley", "rolls royce", "mini",
    # Warning / safety
    "warning light", "check engine", "dashboard light", "recall",
    "road safety", "crash", "collision", "accident",
}

# Questions containing these words are almost certainly off-topic
BLOCK_KEYWORDS = {
    "recipe", "cooking", "bake", "baking", "poem", "poetry",
    "weather", "forecast", "stock market", "cryptocurrency", "bitcoin",
    "politics", "election", "president", "prime minister",
    "movie", "film", "actor", "actress", "celebrity",
    "medical", "medicine", "doctor", "hospital", "disease",
    "law", "lawyer", "legal advice",
    "relationship", "dating", "marriage",
    "sports score", "football score", "basketball score",
    "homework", "essay", "story", "fiction",
    "translate", "translation",
}

# ------------------------------------------------------------------ #
# LLM classification prompt
# ------------------------------------------------------------------ #

CLASSIFY_PROMPT = """\
Is the following question related to vehicles, cars, trucks, motorcycles, \
automotive systems, vehicle maintenance, or driving?

Question: "{question}"

Answer with exactly one word — yes or no:"""


class TopicGuardrail:
    """
    Ensures only vehicle-related questions reach the RAG pipeline.
    Works fully offline — classification uses the local Qwen model.
    """

    BLOCKED_RESPONSE = (
        "I'm sorry, I can only assist with vehicle-related questions such as "
        "maintenance, repairs, parts, diagnostics, and driving. "
        "Please ask me something about vehicles."
    )

    def __init__(self, llm=None) -> None:
        # Lazy-import to avoid circular deps; llm injected or created on demand
        self._llm = llm
        self._vehicle_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in VEHICLE_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )
        self._block_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in BLOCK_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )

    def check(self, question: str) -> Tuple[bool, str]:
        """
        Returns (is_allowed: bool, reason: str).

        reason values:
          "keyword_match"      – allowed by keyword
          "keyword_block"      – blocked by keyword
          "llm_classification" – allowed by LLM
          "llm_block"          – blocked by LLM
          "llm_error"          – LLM failed, defaulted to block
        """
        q = question.strip()

        # Layer 1a — fast allow
        if self._vehicle_re.search(q):
            logger.debug("Guardrail ALLOW (keyword_match): %s", q[:80])
            return True, "keyword_match"

        # Layer 1b — fast block
        if self._block_re.search(q):
            logger.info("Guardrail BLOCK (keyword_block): %s", q[:80])
            return False, "keyword_block"

        # Layer 2 — LLM classification for ambiguous questions
        return self._llm_classify(q)

    def is_allowed(self, question: str) -> bool:
        allowed, _ = self.check(question)
        return allowed

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _llm_classify(self, question: str) -> Tuple[bool, str]:
        try:
            llm = self._get_llm()
            prompt = CLASSIFY_PROMPT.format(question=question)
            response = llm.generate(prompt).strip().lower()
            # Accept "yes", "yes.", "yes!" etc.
            is_vehicle = response.startswith("yes")
            reason = "llm_classification" if is_vehicle else "llm_block"
            logger.info(
                "Guardrail %s (llm_classification) response='%s' question='%s'",
                "ALLOW" if is_vehicle else "BLOCK",
                response[:20],
                question[:80],
            )
            return is_vehicle, reason
        except Exception as exc:
            # Fail safe — block on LLM error
            logger.error("Guardrail LLM classification failed: %s — defaulting to BLOCK", exc)
            return False, "llm_error"

    def _get_llm(self):
        if self._llm is None:
            from core.llm import OllamaLLM
            self._llm = OllamaLLM()
        return self._llm
