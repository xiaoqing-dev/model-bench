"""Judge calibration — the acceptance test for your judge (layer C).

You hand-label pairs you already have an opinion on; this runs the judge on them
and reports two numbers:

  - agreement: how often the judge's swap-tested winner matches YOUR label.
  - position_consistency: how often the two orderings agreed (high = the judge
    isn't just picking by position).

A judge with low agreement is a broken ruler. Fix the rubric or change the judge
model before you trust any leaderboard it produces.

Labelled-set YAML shape:
    rubric_file: rubric.md      # or: rubric: "..."
    pairs:
      - better: "<text you judge better>"
        worse:  "<text you judge worse>"
      - better: "..."
        worse:  "..."
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .client import Client, OpenRouterClient
from .judge import compare_pair


async def calibrate(
    labelled_set_path: str,
    judge_model: str,
    client: Optional[Client] = None,
    **judge_params,
) -> dict:
    client = client or OpenRouterClient()
    spec = yaml.safe_load(Path(labelled_set_path).read_text())
    base = Path(labelled_set_path).parent

    rubric = spec.get("rubric")
    if rubric is None and spec.get("rubric_file"):
        rubric = (base / spec["rubric_file"]).read_text()
    if not rubric:
        raise ValueError("rubric or rubric_file required in labelled set")

    pairs = spec["pairs"]
    rows = []
    agree = 0
    consistent = 0
    for i, pair in enumerate(pairs):
        # Put the known-better text on side A; a correct judge should pick "A".
        v = await compare_pair(
            client, judge_model, rubric, pair["better"], pair["worse"], **judge_params
        )
        is_agree = v.winner == "A"
        agree += int(is_agree)
        consistent += int(v.consistent)
        rows.append(
            {
                "index": i,
                "judge_winner": v.winner,  # "A" means judge agreed with you
                "agreed": is_agree,
                "consistent": v.consistent,
            }
        )

    n = len(pairs)
    return {
        "n": n,
        "agreement": agree / n if n else 0.0,
        "position_consistency": consistent / n if n else 0.0,
        "rows": rows,
    }
