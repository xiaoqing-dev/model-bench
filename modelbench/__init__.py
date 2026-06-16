"""model-bench: compare models and prompt versions, scored by an LLM judge."""

from .schema import PromptVersion, Case, RunResult
from .client import Client, Completion, OpenRouterClient
from .matrix import run_matrix
from .judge import Verdict, PairOutcome, compare_pair, judge_case, judge_matrix
from .aggregate import win_rates
from .generate import generate_system_variants
from .funnel import run_funnel

__all__ = [
    "PromptVersion",
    "Case",
    "RunResult",
    "Client",
    "Completion",
    "OpenRouterClient",
    "run_matrix",
    "Verdict",
    "PairOutcome",
    "compare_pair",
    "judge_case",
    "judge_matrix",
    "win_rates",
    "generate_system_variants",
    "run_funnel",
]
