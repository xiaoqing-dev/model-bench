"""Rule-guided system-prompt variant generation.

You provide a base system prompt and a set of rules (e.g. principles for making
replies more human / warm / fluent). A generator model produces N distinct
variants, each required to apply those rules — not random rewrites. The variants
then go through the normal matrix/funnel + judge to see which one actually works.
"""

from __future__ import annotations

from .client import Client
from .judge import _extract_json  # reuse the tolerant JSON extractor

GENERATE_MAX_TOKENS = 6000

_GEN_INSTRUCTIONS = """You are a senior prompt engineer. Produce {n} distinct variants of a SYSTEM PROMPT.

The system prompt shapes how an assistant replies to end users. The goal of every
variant is to make the assistant's replies more human, warm, natural, and fluent
(有人味 / 有温度 / 流畅), while staying faithful to the base prompt's purpose.

BASE SYSTEM PROMPT:
\"\"\"
{base}
\"\"\"

RULES every variant MUST follow (these are the design principles being tested):
{rules}

Requirements:
- Produce exactly {n} variants that are MEANINGFULLY different from each other —
  vary which rules they emphasise and HOW they operationalise them, not just wording.
- Each variant must genuinely apply the rules above; do not ignore them.
- Keep each variant a usable system prompt (instructions to the assistant), not a
  description of a system prompt.

Output ONLY this JSON, nothing else:
{{"variants": ["<variant 1>", "<variant 2>", ...]}}"""


async def generate_system_variants(
    client: Client,
    model: str,
    base_system: str,
    rules: str,
    n: int = 4,
    **params,
) -> list:
    """Return a list of N system-prompt strings generated from the rules."""
    params.setdefault("max_tokens", GENERATE_MAX_TOKENS)
    prompt = _GEN_INSTRUCTIONS.format(n=n, base=base_system, rules=rules)
    comp = await client.complete(model, prompt, **params)
    data = _extract_json(comp.text)
    variants = [str(v) for v in data.get("variants", []) if str(v).strip()]
    return variants[:n]
