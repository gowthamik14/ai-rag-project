"""
Abstract base class for all workflows.

Every workflow:
  1. Receives an input dict
  2. Validates it
  3. Executes a sequence of named steps
  4. Returns a WorkflowResult

Subclasses implement `_steps` (ordered list of step methods) and `_validate`.
"""
from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class WorkflowResult:
    workflow: str
    status: WorkflowStatus
    output: Dict[str, Any] = field(default_factory=dict)
    steps_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow,
            "status": self.status.value,
            "output": self.output,
            "steps_completed": self.steps_completed,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 3),
        }


class BaseWorkflow(ABC):
    """
    Template-method pattern workflow.

    Subclasses override:
      name        – human-readable workflow name
      _validate   – raise ValueError if inputs are wrong
      _steps      – property returning ordered list of (step_name, callable)
    """

    name: str = "base"

    def run(self, inputs: Dict[str, Any]) -> WorkflowResult:
        """Entry point: validate → execute steps → return result."""
        start = time.perf_counter()
        result = WorkflowResult(workflow=self.name, status=WorkflowStatus.SUCCESS)
        ctx: Dict[str, Any] = {"inputs": inputs}

        # Validate
        try:
            self._validate(inputs)
        except ValueError as exc:
            result.status = WorkflowStatus.FAILED
            result.errors.append(f"Validation error: {exc}")
            result.duration_seconds = time.perf_counter() - start
            logger.error("[%s] validation failed: %s", self.name, exc)
            return result

        # Execute steps
        for step_name, step_fn in self._steps:
            logger.info("[%s] ▶ step: %s", self.name, step_name)
            try:
                step_fn(ctx)
                result.steps_completed.append(step_name)
            except Exception as exc:
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"{step_name}: {exc}")
                logger.error(
                    "[%s] step '%s' failed: %s\n%s",
                    self.name,
                    step_name,
                    exc,
                    traceback.format_exc(),
                )
                break  # stop on first failure

        result.output = ctx.get("output", {})
        result.duration_seconds = time.perf_counter() - start
        logger.info(
            "[%s] finished | status=%s | duration=%.3fs",
            self.name,
            result.status.value,
            result.duration_seconds,
        )
        return result

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _validate(self, inputs: Dict[str, Any]) -> None:
        """Raise ValueError with a descriptive message on invalid input."""
        ...

    @property
    @abstractmethod
    def _steps(self) -> List[tuple]:
        """Return list of (step_name: str, step_fn: Callable[[ctx], None])."""
        ...
