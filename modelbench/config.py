"""Load an experiment from YAML and run it end to end."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .aggregate import win_rates
from .client import Client, OpenRouterClient
from .judge import judge_matrix
from .matrix import run_matrix
from .schema import Case, PromptVersion


@dataclass
class Experiment:
    prompts: list
    models: list
    cases: list
    judge_model: str
    rubric: str
    axis: str = "model"  # "model" or "prompt"
    params: Optional[dict] = None
    judge_params: Optional[dict] = None


def load_experiment(path: str) -> Experiment:
    spec = yaml.safe_load(Path(path).read_text())
    base = Path(path).parent

    prompts = [PromptVersion(id=p["id"], template=p["template"]) for p in spec["prompts"]]
    cases = [
        Case(id=c["id"], vars=c.get("vars", {}), reference=c.get("reference"))
        for c in spec["cases"]
    ]
    judge = spec["judge"]
    rubric = judge.get("rubric")
    if rubric is None and judge.get("rubric_file"):
        rubric = (base / judge["rubric_file"]).read_text()
    if not rubric:
        raise ValueError("judge.rubric or judge.rubric_file is required")

    return Experiment(
        prompts=prompts,
        models=list(spec["models"]),
        cases=cases,
        judge_model=judge["model"],
        rubric=rubric,
        axis=spec.get("axis", "model"),
        params=spec.get("params"),
        judge_params=judge.get("params"),
    )


async def run_experiment(exp: Experiment, client: Optional[Client] = None) -> dict:
    """Run the matrix, judge it, and return {results, outcomes, standings}."""
    client = client or OpenRouterClient()
    results = await run_matrix(
        exp.prompts, exp.models, exp.cases, client, params=exp.params or {}
    )
    outcomes = await judge_matrix(
        client,
        exp.judge_model,
        exp.rubric,
        results,
        axis=exp.axis,
        **(exp.judge_params or {}),
    )
    return {
        "results": results,
        "outcomes": outcomes,
        "standings": win_rates(outcomes),
    }
