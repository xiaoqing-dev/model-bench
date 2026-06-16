"""Model-calling layer. Everything goes through OpenRouter's OpenAI-compatible
endpoint, so one async client reaches every model by slug (e.g. "openai/gpt-5",
"anthropic/claude-opus-4.8", "google/gemini-2.5-pro").

`Client` is a Protocol so tests can swap in a fake with no network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

# Load a project-local .env if present, so the key works regardless of which
# shell/terminal/IDE launched the process. Real env vars still take precedence.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # dotenv optional; env var / explicit api_key still work
    pass

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class Completion:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


@runtime_checkable
class Client(Protocol):
    async def complete(self, model: str, prompt: str, **params) -> Completion: ...


class OpenRouterClient:
    """Async client over OpenRouter. Reads OPENROUTER_API_KEY by default."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        default_params: Optional[dict] = None,
        app_title: str = "model-bench",
    ) -> None:
        from openai import AsyncOpenAI  # imported lazily so tests don't need it

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key. Set OPENROUTER_API_KEY in your environment, "
                "or pass api_key=... to OpenRouterClient()."
            )
        self._base_url = base_url
        self._key = key
        self._client = AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            default_headers={"X-Title": app_title},
        )
        self.default_params = default_params or {}

    async def complete(self, model: str, prompt: str, **params) -> Completion:
        merged = {**self.default_params, **params}
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # Ask OpenRouter to include exact $ cost in the usage block.
            extra_body={"usage": {"include": True}},
            **merged,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        pt = ct = cost = None
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            cost = getattr(usage, "cost", None)
            if cost is None:
                extra = getattr(usage, "model_extra", None) or {}
                cost = extra.get("cost")
        return Completion(text=text, prompt_tokens=pt, completion_tokens=ct, cost_usd=cost)

    async def list_models(self) -> list:
        """Fetch the live model catalog (slugs + pricing). Don't hardcode slugs —
        they change. Returns the raw OpenRouter `data` list."""
        import httpx

        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._key}"},
            )
            r.raise_for_status()
            return r.json().get("data", [])
