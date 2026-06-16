import json

from modelbench.client import Completion
from modelbench.generate import generate_system_variants


class GenClient:
    def __init__(self, variants):
        self.variants = variants
        self.last_prompt = None

    async def complete(self, model, prompt, *, system="", **params):
        self.last_prompt = prompt
        # simulate a model wrapping JSON in some prose
        return Completion(text="Sure!\n" + json.dumps({"variants": self.variants}))


async def test_generate_returns_variants():
    client = GenClient(["A", "B", "C", "D"])
    out = await generate_system_variants(client, "gen", "base prompt", "rule1\nrule2", n=4)
    assert out == ["A", "B", "C", "D"]


async def test_generate_truncates_to_n():
    client = GenClient(["A", "B", "C", "D"])
    out = await generate_system_variants(client, "gen", "base", "rules", n=2)
    assert out == ["A", "B"]


async def test_generate_drops_blank_variants():
    client = GenClient(["A", "  ", "B"])
    out = await generate_system_variants(client, "gen", "base", "rules", n=4)
    assert out == ["A", "B"]


async def test_generate_includes_rules_in_prompt():
    client = GenClient(["A"])
    await generate_system_variants(client, "gen", "BASE_TEXT", "RULE_TEXT", n=1)
    assert "BASE_TEXT" in client.last_prompt
    assert "RULE_TEXT" in client.last_prompt
