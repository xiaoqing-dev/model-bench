"""Run the (prompt x model x case) matrix concurrently.

Failure is isolated per cell: if one model errors or times out, its RunResult
carries the error string and every other cell still completes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterable, Optional

from .client import Client
from .schema import Case, PromptVersion, RunResult
from .templating import render


async def run_matrix(
    prompts: Iterable[PromptVersion],
    models: Iterable[str],
    cases: Iterable[Case],
    client: Client,
    *,
    concurrency: int = 8,
    params: Optional[dict] = None,
) -> list:
    """Execute every combination and return a flat list of RunResult.

    `concurrency` caps how many model calls are in flight at once.
    `params` (e.g. {"temperature": 0.7, "max_tokens": 1024}) is passed to every call.
    """
    params = params or {}
    prompts = list(prompts)
    models = list(models)
    cases = list(cases)
    sem = asyncio.Semaphore(concurrency)

    async def run_one(p: PromptVersion, m: str, c: Case) -> RunResult:
        async with sem:
            t0 = time.perf_counter()
            try:
                rendered = render(p.template, c.vars)
                comp = await client.complete(m, rendered, **params)
                # Empty content with no exception is the reasoning-model trap:
                # the response is "successful" but the answer is blank (hidden
                # reasoning ate the whole max_tokens). Surface it, don't drop it.
                error = None
                if not (comp.text and comp.text.strip()):
                    if comp.finish_reason == "length":
                        error = "空输出:max_tokens 被模型推理耗尽,请调高 max_tokens"
                    else:
                        error = f"空输出(finish={comp.finish_reason})"
                return RunResult(
                    prompt_id=p.id,
                    model=m,
                    case_id=c.id,
                    output=comp.text,
                    error=error,
                    latency_s=time.perf_counter() - t0,
                    prompt_tokens=comp.prompt_tokens,
                    completion_tokens=comp.completion_tokens,
                    cost_usd=comp.cost_usd,
                )
            except Exception as e:  # isolate: one bad cell never sinks the batch
                return RunResult(
                    prompt_id=p.id,
                    model=m,
                    case_id=c.id,
                    error=f"{type(e).__name__}: {e}",
                    latency_s=time.perf_counter() - t0,
                )

    tasks = [run_one(p, m, c) for p in prompts for m in models for c in cases]
    return await asyncio.gather(*tasks)
