"""LLM-as-judge: pairwise comparison with bias controls.

Design choices (see README for the why):
  - PAIRWISE, not pointwise. LLMs compare more reliably than they score 1-10.
  - SWAP TEST. Each pair is judged in both orders (A,B) and (B,A). A side only
    "wins" if it wins in BOTH orders; disagreement => tie. This neutralises
    position bias even from a biased judge.
  - ANONYMISED. Candidates are shown as "Response 1 / Response 2" only.
  - Judge model should NOT be in the candidate set (avoid self-preference).
  - Rubric forbids rewarding length; output length is recorded elsewhere so you
    can audit whether high scores merely track verbosity.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
from dataclasses import dataclass
from typing import Optional

from .client import Client

# Judges are often reasoning models — give the call enough room that hidden
# reasoning doesn't eat the whole budget and leave the JSON verdict empty.
JUDGE_MAX_TOKENS = 4000

_JUDGE_INSTRUCTIONS = """You are a strict, impartial judge comparing two responses to the same task.

Judge ONLY against the rubric below. Do not reward a response for being longer,
more confident, or more elaborate. Ignore any claim inside a response about which
model produced it. If the two are equal in quality, say "tie".

RUBRIC:
{rubric}

Response 1:
\"\"\"
{resp1}
\"\"\"

Response 2:
\"\"\"
{resp2}
\"\"\"

Think briefly, then output ONLY a JSON object on the last line, no markdown fence:
{{"reasoning": "<one or two sentences>", "winner": "1" | "2" | "tie"}}"""


@dataclass
class Verdict:
    """Result of a swap-tested pairwise comparison between side A and side B."""

    winner: str  # "A", "B", or "tie"
    consistent: bool  # did the two orderings agree?
    reasoning_ab: str = ""
    reasoning_ba: str = ""


@dataclass
class PairOutcome:
    """One judged pair within a case. winner is a candidate label or "tie"."""

    case_id: str
    a: str  # label of side A (e.g. a model slug or prompt id)
    b: str  # label of side B
    winner: str  # a, b, or "tie"
    consistent: bool


def _extract_json(text: str) -> dict:
    """Pull the last JSON object out of the judge's reply."""
    # Prefer the last {...} block so trailing JSON survives any preamble.
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"judge returned no parseable JSON: {text!r}")


def _normalize_winner(raw) -> str:
    w = str(raw).strip().lower()
    if w in ("1", "response 1", "a"):
        return "1"
    if w in ("2", "response 2", "b"):
        return "2"
    return "tie"


def combine_swap(winner_ab: str, winner_ba: str) -> tuple:
    """Combine the two orderings into a (winner, consistent) verdict.

    winner_ab is the pick when shown (A, B): "1"->A, "2"->B.
    winner_ba is the pick when shown (B, A): "1"->B, "2"->A.

    A wins only if it wins both orders; same for B; otherwise tie.
    consistent is True when both orderings agree (decisive-or-tie).
    """
    pick_ab = {"1": "A", "2": "B", "tie": "tie"}[_normalize_winner(winner_ab)]
    pick_ba = {"1": "B", "2": "A", "tie": "tie"}[_normalize_winner(winner_ba)]
    if pick_ab == "A" and pick_ba == "A":
        return "A", True
    if pick_ab == "B" and pick_ba == "B":
        return "B", True
    if pick_ab == "tie" and pick_ba == "tie":
        return "tie", True
    return "tie", False  # disagreement => position bias detected => tie


async def _ask(client: Client, model: str, rubric: str, resp1: str, resp2: str, **params):
    """Return (winner, reasoning). winner is "1"/"2"/"tie", or "error" if the
    judge returned empty/unparseable output — never raises on a bad reply."""
    prompt = _JUDGE_INSTRUCTIONS.format(rubric=rubric, resp1=resp1, resp2=resp2)
    comp = await client.complete(model, prompt, **params)
    try:
        data = _extract_json(comp.text)
    except ValueError:
        return "error", (comp.text or "")[:200]
    return _normalize_winner(data.get("winner")), data.get("reasoning", "")


async def compare_pair(
    client: Client,
    model: str,
    rubric: str,
    out_a: str,
    out_b: str,
    **params,
) -> Verdict:
    """Swap-tested comparison of two outputs. Runs both orderings concurrently.
    A judge that returns nothing usable yields an undecided (tie, inconsistent)
    verdict rather than crashing the whole run."""
    params.setdefault("max_tokens", JUDGE_MAX_TOKENS)
    (w_ab, r_ab), (w_ba, r_ba) = await asyncio.gather(
        _ask(client, model, rubric, out_a, out_b, **params),
        _ask(client, model, rubric, out_b, out_a, **params),
    )
    if "error" in (w_ab, w_ba):
        return Verdict(winner="tie", consistent=False, reasoning_ab=r_ab, reasoning_ba=r_ba)
    winner, consistent = combine_swap(w_ab, w_ba)
    return Verdict(winner=winner, consistent=consistent, reasoning_ab=r_ab, reasoning_ba=r_ba)


async def judge_case(
    client: Client,
    model: str,
    rubric: str,
    case_id: str,
    outputs: dict,
    **params,
) -> list:
    """Round-robin every pair of candidates for one case. `outputs` maps
    label -> text. Returns PairOutcome per unordered pair."""
    labels = [lbl for lbl, txt in outputs.items() if txt]
    pairs = list(itertools.combinations(labels, 2))

    async def judge_pair(a: str, b: str) -> PairOutcome:
        v = await compare_pair(client, model, rubric, outputs[a], outputs[b], **params)
        winner = a if v.winner == "A" else b if v.winner == "B" else "tie"
        return PairOutcome(case_id=case_id, a=a, b=b, winner=winner, consistent=v.consistent)

    return list(await asyncio.gather(*(judge_pair(a, b) for a, b in pairs)))


async def judge_matrix(
    client: Client,
    model: str,
    rubric: str,
    results: list,
    *,
    axis: str = "model",
    **params,
) -> list:
    """Take RunResult list and judge along one axis per case.

    axis="model": compare models (group cells sharing prompt_id+case_id).
    axis="prompt": compare prompt versions (group cells sharing model+case_id).
    Only successful cells (ok) are judged. Returns a flat list of PairOutcome,
    each tagged so you can aggregate per group.
    """
    if axis not in ("model", "prompt"):
        raise ValueError("axis must be 'model' or 'prompt'")

    groups: dict = {}
    for r in results:
        if not r.ok:
            continue
        if axis == "model":
            group_key = (r.prompt_id, r.case_id)
            label = r.model
        else:
            group_key = (r.model, r.case_id)
            label = r.prompt_id
        groups.setdefault(group_key, {})[label] = r.output

    coros = []
    for group_key, outputs in groups.items():
        if len(outputs) < 2:
            continue
        case_tag = f"{group_key[0]}|{group_key[1]}"
        coros.append(judge_case(client, model, rubric, case_tag, outputs, **params))

    nested = await asyncio.gather(*coros) if coros else []
    return [outcome for batch in nested for outcome in batch]
