"""Funnel (tournament) runner — axis-agnostic.

Round 1 (scout): run every candidate on a small subset of cases, judge, rank.
Promote the top-K. Round 2 (deep): run only the survivors on the full case set
and judge again. Saves money by culling weak candidates before the expensive
deep comparison.

axis="model"  -> candidates are models (cull the model field).
axis="prompt" -> candidates are prompt/system-variant versions (cull variants).
"""

from __future__ import annotations

from typing import Optional

from .aggregate import win_rates
from .client import Client
from .judge import judge_matrix
from .matrix import run_matrix


def _default_scout_cases(cases: list) -> list:
    """Use ~half the cases (at least 1) for the cheap scouting round."""
    if len(cases) <= 1:
        return list(cases)
    return list(cases[: max(1, len(cases) // 2)])


async def run_funnel(
    prompts: list,
    models: list,
    cases: list,
    client: Client,
    judge_model: str,
    rubric: str,
    *,
    axis: str = "model",
    top_k: int = 3,
    scout_cases: Optional[list] = None,
    params: Optional[dict] = None,
    judge_params: Optional[dict] = None,
) -> dict:
    if axis not in ("model", "prompt"):
        raise ValueError("axis must be 'model' or 'prompt'")
    params = params or {}
    judge_params = judge_params or {}
    scout = scout_cases if scout_cases is not None else _default_scout_cases(cases)

    # ---- Round 1: scout on a subset, rank, promote top_k ----
    r1 = await run_matrix(prompts, models, scout, client, params=params)
    o1 = await judge_matrix(client, judge_model, rubric, r1, axis=axis, **judge_params)
    s1 = win_rates(o1)
    promoted = [s.label for s in s1[:top_k]]

    # If judging produced no ranking (e.g. <2 candidates), promote everyone present.
    if not promoted:
        promoted = models if axis == "model" else [p.id for p in prompts]

    # ---- Round 2: deep run on full cases, survivors only ----
    if axis == "model":
        deep_models = [m for m in models if m in promoted]
        deep_prompts = prompts
    else:
        deep_models = models
        deep_prompts = [p for p in prompts if p.id in promoted]

    r2 = await run_matrix(deep_prompts, deep_models, cases, client, params=params)
    o2 = await judge_matrix(client, judge_model, rubric, r2, axis=axis, **judge_params)
    s2 = win_rates(o2)

    return {
        "promoted": promoted,
        "round1": {"results": r1, "outcomes": o1, "standings": s1},
        "round2": {"results": r2, "outcomes": o2, "standings": s2},
    }
