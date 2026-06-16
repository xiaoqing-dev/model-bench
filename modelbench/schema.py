"""Core data structures. One unifying idea:

    a run = (prompt_version, model, case) -> output + metadata

Hold different axes fixed and you get different comparisons:
  - fix prompt+case, vary model   -> model comparison
  - fix model+case, vary prompt   -> prompt A/B test
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PromptVersion:
    """A prompt version. `template` is the user message (may contain {{var}}).
    `system` is an optional system prompt — this is what you tune when comparing
    system-prompt variants."""

    id: str
    template: str
    system: str = ""


@dataclass(frozen=True)
class Case:
    """One test input: variable bindings for a template, plus optional reference."""

    id: str
    vars: dict = field(default_factory=dict)
    reference: Optional[str] = None


@dataclass
class RunResult:
    """The output of one (prompt, model, case) cell, with metadata."""

    prompt_id: str
    model: str
    case_id: str
    output: Optional[str] = None
    latency_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def key(self) -> tuple:
        return (self.prompt_id, self.model, self.case_id)
