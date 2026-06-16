"""Fakes that let the whole pipeline run offline, with no API key and no network.

These are the heart of layer-A testing: they make the deterministic logic
(matrix fan-out, failure isolation, swap-test combination, aggregation)
verifiable without touching a real model.
"""

from __future__ import annotations

import json

from modelbench.client import Completion


class FakeModelClient:
    """Deterministic stand-in for a model client.

    Output is "<model>::<prompt>". Models listed in `fail_models` raise, so we
    can test failure isolation.
    """

    def __init__(self, fail_models=(), empty_models=()):
        self.fail_models = set(fail_models)
        self.empty_models = set(empty_models)  # return blank content (reasoning trap)
        self.calls = []

    async def complete(self, model: str, prompt: str, *, system: str = "", **params) -> Completion:
        self.calls.append((model, prompt, system))
        if model in self.fail_models:
            raise RuntimeError("simulated provider failure")
        if model in self.empty_models:
            return Completion(text="", cost_usd=0.001, finish_reason="length")
        return Completion(
            text=f"{model}::{prompt}",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.001,
            finish_reason="stop",
        )


class ScriptedJudge:
    """A judge whose verdict is a pure function of the texts it sees, so judge
    behaviour is deterministic in tests.

    mode="prefer_token": picks whichever response contains `token`; "tie" if
        neither or both do. A *good* judge — order-independent.
    mode="always_first": always picks Response 1, ignoring content. A *position-
        biased* judge — used to prove the swap test neutralises bias.
    """

    def __init__(self, mode="prefer_token", token="GOOD"):
        self.mode = mode
        self.token = token
        self.calls = []

    async def complete(self, model: str, prompt: str, *, system: str = "", **params) -> Completion:
        self.calls.append((model, prompt))
        resp1, resp2 = _extract_two_responses(prompt)
        if self.mode == "empty":
            return Completion(text="")  # judge returned nothing usable
        if self.mode == "always_first":
            winner = "1"
        else:
            in1 = self.token in resp1
            in2 = self.token in resp2
            if in1 and not in2:
                winner = "1"
            elif in2 and not in1:
                winner = "2"
            else:
                winner = "tie"
        return Completion(text=json.dumps({"reasoning": "scripted", "winner": winner}))


class FakeBench:
    """One client that acts as candidate models AND as the judge, routed by model
    name — needed for funnel/experiment tests where run_matrix and judge_matrix
    share a single client. Candidates output "<model>::<prompt>"; the judge picks
    whichever response contains `token`."""

    def __init__(self, judge_model: str = "judge", token: str = "GOOD"):
        self.judge_model = judge_model
        self.token = token
        self.calls = []

    async def complete(self, model: str, prompt: str, *, system: str = "", **params) -> Completion:
        self.calls.append((model, prompt, system))
        if model == self.judge_model:
            r1, r2 = _extract_two_responses(prompt)
            in1, in2 = self.token in r1, self.token in r2
            winner = "1" if in1 and not in2 else "2" if in2 and not in1 else "tie"
            return Completion(text=json.dumps({"reasoning": "scripted", "winner": winner}))
        return Completion(text=f"{model}::{prompt}", cost_usd=0.001, finish_reason="stop")


def _extract_two_responses(prompt: str):
    """Pull the two triple-quoted blocks out of the judge prompt."""
    parts = prompt.split('"""')
    # blocks are at odd indices: [pre, resp1, mid, resp2, post]
    blocks = [parts[i] for i in range(1, len(parts), 2)]
    resp1 = blocks[0] if len(blocks) > 0 else ""
    resp2 = blocks[1] if len(blocks) > 1 else ""
    return resp1, resp2
